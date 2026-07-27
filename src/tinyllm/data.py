"""Stage 1 -- corpus streaming, tokenization, and packing.

The output of this stage is a flat ``uint16`` token stream on disk. That format
is chosen deliberately:

  * **uint16** because an 8192-entry vocabulary fits in 16 bits. Storing 328M
    tokens as int64 would be 2.6 GB; as uint16 it is 656 MB.
  * **flat, not per-document** because the training loop samples random windows
    of ``seq_len + 1`` from a single contiguous array. No padding, no ragged
    batches, no wasted compute. Documents are separated by EOS, so windows that
    straddle a boundary teach the model what "end of story" looks like.
  * **memory-mapped** so the loader never holds the corpus in RAM.

The dataset is streamed rather than downloaded: TinyStories is ~7.6 GB on disk
and Colab's quota is not worth spending on data we read exactly once.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, Iterator

import numpy as np

from tinyllm.config import data_cfg
from tinyllm.tokenizer import encode_stream

TOKEN_DTYPE = np.uint16


# ---------------------------------------------------------------------------
# Streaming source
# ---------------------------------------------------------------------------

def stream_texts(
    dataset_id: str = data_cfg.dataset_id,
    split: str = data_cfg.train_split,
    limit: int | None = None,
    text_field: str = "text",
) -> Iterator[str]:
    """Yield raw documents from a Hub dataset without downloading it.

    ``limit`` caps the number of documents -- used by the smoke run so the
    whole pipeline can be rehearsed in under a minute.
    """
    from datasets import load_dataset

    ds = load_dataset(dataset_id, split=split, streaming=True)
    for i, row in enumerate(ds):
        if limit is not None and i >= limit:
            break
        text = row.get(text_field)
        if text:
            yield text


def corpus_stats(texts: Iterable[str], sample: int = 20_000) -> dict:
    """Document-length statistics, for the stage 1 write-up.

    Worth looking at before training: it tells you what fraction of documents
    fit inside the context window, which bounds how much long-range structure
    the model can possibly learn.
    """
    char_lens: list[int] = []
    word_lens: list[int] = []
    for i, t in enumerate(texts):
        if i >= sample:
            break
        char_lens.append(len(t))
        word_lens.append(len(t.split()))

    if not char_lens:
        return {"n": 0}

    ca = np.array(char_lens)
    wa = np.array(word_lens)
    return {
        "n_sampled": len(char_lens),
        "chars_mean": float(ca.mean()),
        "chars_p50": float(np.percentile(ca, 50)),
        "chars_p95": float(np.percentile(ca, 95)),
        "chars_max": int(ca.max()),
        "words_mean": float(wa.mean()),
        "words_p50": float(np.percentile(wa, 50)),
        "words_p95": float(np.percentile(wa, 95)),
    }


# ---------------------------------------------------------------------------
# Tokenize and pack
# ---------------------------------------------------------------------------

def pack_tokens(
    sp,
    texts: Iterable[str],
    out_path: Path,
    max_tokens: int | None = None,
    log_every: int = 5_000_000,
) -> dict:
    """Tokenize documents and append them to a flat uint16 file.

    Each document is wrapped in BOS/EOS. Returns a stats dict that is saved
    alongside the .bin so a token file is never ambiguous about what produced it.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)

    total = 0
    n_docs = 0
    next_log = log_every
    buf: list[int] = []
    BUF_FLUSH = 1_000_000

    with out_path.open("wb") as f:
        for ids in encode_stream(sp, texts, add_bos=True, add_eos=True):
            buf.extend(ids)
            total += len(ids)
            n_docs += 1

            if len(buf) >= BUF_FLUSH:
                np.array(buf, dtype=TOKEN_DTYPE).tofile(f)
                buf.clear()

            if total >= next_log:
                print(f"    {total / 1e6:6.1f}M tokens / {n_docs:,} docs")
                next_log += log_every

            if max_tokens is not None and total >= max_tokens:
                break

        if buf:
            np.array(buf, dtype=TOKEN_DTYPE).tofile(f)

    stats = {
        "path": str(out_path),
        "n_tokens": total,
        "n_docs": n_docs,
        "tokens_per_doc": total / max(1, n_docs),
        "dtype": str(np.dtype(TOKEN_DTYPE)),
        "size_mb": out_path.stat().st_size / 1e6,
    }
    out_path.with_suffix(".json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
    return stats


def load_tokens(path: Path, in_memory: bool = True) -> np.ndarray:
    """Load a packed token file.

    ``in_memory=True`` reads the whole array into RAM. The training corpus is
    ~660 MB as uint16 and Colab has ~12 GB, so this is affordable -- and it
    matters: the loader draws 128 random 512-token windows per optimizer step,
    and serving those from a memory-mapped file on Colab's disk is slow enough
    to dominate step time. Set False if the corpus outgrows RAM.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"{path} not found -- run pack_tokens() first (notebook 01).")
    if in_memory:
        return np.fromfile(path, dtype=TOKEN_DTYPE)
    return np.memmap(path, dtype=TOKEN_DTYPE, mode="r")


def ensure_token_file(name: str, data_dir: Path, repo_id: str) -> Path:
    """Return the path to a packed token file, fetching it from the Hub if the
    runtime doesn't have it.

    Colab recycles runtimes, so ``/content/work`` is empty at the start of every
    session after the first. Notebooks 05-07 need ``val.bin`` and would
    otherwise silently bind it to None and fail several cells later with a
    confusing TypeError.
    """
    from huggingface_hub import hf_hub_download

    data_dir = Path(data_dir)
    local = data_dir / name
    if local.exists():
        return local

    try:
        src = hf_hub_download(repo_id=repo_id, filename=f"data/{name}")
    except Exception as e:
        raise FileNotFoundError(
            f"{name} is not on this runtime and could not be fetched from "
            f"{repo_id}: {type(e).__name__}: {e}\n"
            "Run section 2.7 of notebook 02 to upload the packed token files, "
            "or re-run notebook 02 to rebuild them."
        ) from e

    data_dir.mkdir(parents=True, exist_ok=True)
    local.write_bytes(Path(src).read_bytes())
    return local


def causal_loss(logits, targets):
    """Next-token cross-entropy against the pre-shifted targets from get_batch.

    Use this rather than passing ``labels=`` to a HuggingFace model.

    ``LlamaForCausalLM`` shifts labels internally::

        shift_logits = logits[..., :-1, :]
        shift_labels = labels[..., 1:]

    and ``get_batch`` already returns ``y`` shifted by one. Passing ``labels=y``
    therefore shifts twice and trains the model to predict the token *two*
    positions ahead. That converges to a perfectly healthy-looking loss curve
    and produces incoherent text, because generation samples the next token from
    a distribution that was fitted for the one after it.

    Computing the loss here makes the convention explicit and uses all
    ``seq_len`` predictions instead of discarding the last one.
    """
    import torch.nn.functional as F

    # reshape, not view: callers pass slices such as logits[:, :-1, :], which
    # are non-contiguous, and view() rejects those.
    return F.cross_entropy(
        logits.float().reshape(-1, logits.size(-1)),
        targets.reshape(-1),
    )


def describe(path: Path) -> dict:
    """Read back the sidecar stats written by pack_tokens."""
    meta = Path(path).with_suffix(".json")
    if meta.exists():
        return json.loads(meta.read_text(encoding="utf-8"))
    arr = load_tokens(path)
    return {"path": str(path), "n_tokens": int(arr.shape[0])}


# ---------------------------------------------------------------------------
# Batching
# ---------------------------------------------------------------------------

def get_batch(
    tokens: np.ndarray,
    batch_size: int,
    seq_len: int,
    device: str = "cuda",
    rng: np.random.Generator | None = None,
):
    """Sample a random batch of contiguous windows.

    ``y`` is ``x`` shifted one position: at every position the model predicts
    the next token, so a single forward pass yields ``seq_len`` predictions
    rather than one. That is what makes causal LM training efficient.

    Pinned memory + ``non_blocking`` lets the H2D copy overlap compute, which is
    worth a few percent on a T4 where the model is small enough that data
    movement is a visible fraction of step time.
    """
    import torch

    rng = rng or np.random.default_rng()
    max_start = len(tokens) - seq_len - 1
    if max_start <= 0:
        raise ValueError(
            f"Token file has {len(tokens):,} tokens, too few for seq_len={seq_len}."
        )

    starts = rng.integers(0, max_start, size=batch_size)
    # .astype(np.int64) copies out of the memmap; torch cannot use uint16 directly.
    x = np.stack([tokens[s : s + seq_len] for s in starts]).astype(np.int64)
    y = np.stack([tokens[s + 1 : s + 1 + seq_len] for s in starts]).astype(np.int64)

    xt = torch.from_numpy(x)
    yt = torch.from_numpy(y)
    if device.startswith("cuda"):
        xt = xt.pin_memory().to(device, non_blocking=True)
        yt = yt.pin_memory().to(device, non_blocking=True)
    else:
        xt, yt = xt.to(device), yt.to(device)
    return xt, yt


def iter_eval_batches(
    tokens: np.ndarray,
    batch_size: int,
    seq_len: int,
    n_batches: int,
    device: str = "cuda",
    seed: int = data_cfg.seed,
):
    """Deterministic batches for validation.

    Fixed seed on purpose: comparing val loss across steps only means something
    if every evaluation sees the same windows. A fresh random draw each time
    would add noise that looks like signal.
    """
    rng = np.random.default_rng(seed)
    for _ in range(n_batches):
        yield get_batch(tokens, batch_size, seq_len, device=device, rng=rng)


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def prepare_corpus(
    sp,
    out_dir: Path,
    train_tokens: int | None = None,
    val_tokens: int = data_cfg.val_tokens,
    train_doc_limit: int | None = None,
    smoke: bool = False,
) -> dict:
    """Build train.bin and val.bin. This is the whole of stage 1.

    The validation split comes from the dataset's own held-out split, not a
    slice of train, so there is no chance of contamination inflating the
    perplexity numbers in stage 5.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if smoke:
        train_doc_limit = train_doc_limit or 2_000
        val_tokens = min(val_tokens, 100_000)

    print("  packing validation split")
    val_stats = pack_tokens(
        sp,
        stream_texts(split=data_cfg.val_split),
        out_dir / "val.bin",
        max_tokens=val_tokens,
    )
    print(f"    val:   {val_stats['n_tokens']:,} tokens ({val_stats['size_mb']:.1f} MB)")

    print("  packing training split")
    train_stats = pack_tokens(
        sp,
        stream_texts(split=data_cfg.train_split, limit=train_doc_limit),
        out_dir / "train.bin",
        max_tokens=train_tokens,
    )
    print(f"    train: {train_stats['n_tokens']:,} tokens ({train_stats['size_mb']:.1f} MB)")

    return {"train": train_stats, "val": val_stats}
