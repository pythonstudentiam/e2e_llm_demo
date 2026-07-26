"""Stage 2 -- tokenizer training, wrapping, and round-trip verification.

Two tokenizer objects appear in this project and they have different jobs:

  * ``sentencepiece.SentencePieceProcessor`` -- the fast C++ encoder. Used for
    bulk-tokenizing 328M training tokens in stage 1/4. Nothing else is fast
    enough to be worth using there.

  * ``transformers.LlamaTokenizer`` -- the HF wrapper. Used only for packaging:
    it carries the chat template, the special-token metadata, and the
    ``tokenizer.model`` file that llama.cpp needs. Never used in a hot loop.

Why SentencePiece rather than the `tokenizers` library
------------------------------------------------------
``convert_hf_to_gguf.py`` resolves a Llama model's vocabulary like this::

    try:
        self._set_vocab_sentencepiece()      # <- needs tokenizer.model
    except FileNotFoundError:
        try:
            self._set_vocab_llama_hf()
        except (FileNotFoundError, TypeError):
            self._set_vocab_gpt2()           # <- hashes the pre-tokenizer

The last path compares a hash of the pre-tokenizer against a hardcoded registry
and raises ``NotImplementedError: BPE pre-tokenizer was not recognized`` for
anything it has not seen -- which is every custom `tokenizers` BPE ever trained.
Because ``_set_vocab_sentencepiece()`` is tried *first* and does no hash lookup,
simply shipping a ``tokenizer.model`` file sidesteps the whole problem.

That single file is the difference between stage 8 working and stage 8 failing.
See docs/02-tokenizer.md for the failure mode and how to fix it the hard way.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Iterable, Iterator

import sentencepiece as spm

from tinyllm.config import tok_cfg


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def write_corpus_sample(
    texts: Iterable[str],
    out_path: Path,
    max_sentences: int = tok_cfg.train_sentences,
) -> int:
    """Write one training example per line for the SentencePiece trainer.

    Newlines inside a story are collapsed to spaces because the trainer treats
    each line as an independent sentence; a story split across lines would be
    learned as several unrelated fragments.

    Returns the number of lines written.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with out_path.open("w", encoding="utf-8") as f:
        for text in texts:
            if n >= max_sentences:
                break
            flat = " ".join(text.split())
            if not flat:
                continue
            f.write(flat + "\n")
            n += 1
    return n


def train_sentencepiece(
    corpus_file: Path,
    out_dir: Path,
    vocab_size: int = tok_cfg.vocab_size,
    prefix_name: str = "tokenizer",
) -> Path:
    """Fit a BPE vocabulary. Returns the path to tokenizer.model.

    ``vocab_size`` is a parameter rather than read straight from the config so
    the vocab-size ablation in notebook 02 -- and the unit test in
    tests/test_tokenizer.py -- can train small models cheaply.

    The non-default options all exist to make the tokenizer behave like Llama's
    and to make round-tripping exact:

      byte_fallback              any byte is representable, so nothing is ever
                                 lossy -- costs 256 vocab slots, worth it
      normalization_rule_name    'identity' disables NFKC, so decode(encode(s))
                                 returns *s*, not a normalized variant
      remove_extra_whitespaces   False, same reason
      split_digits               digits tokenize individually, which is what
                                 stops "1997" and "1998" being unrelated atoms
      user_defined_symbols       ChatML control tokens, kept atomic
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    prefix = out_dir / prefix_name

    spm.SentencePieceTrainer.train(
        input=str(corpus_file),
        model_prefix=str(prefix),
        vocab_size=vocab_size,
        model_type=tok_cfg.model_type,
        character_coverage=tok_cfg.character_coverage,
        max_sentence_length=tok_cfg.max_sentence_length,
        input_sentence_size=tok_cfg.train_sentences,
        shuffle_input_sentence=True,
        # Special tokens, in the id order llama.cpp expects.
        unk_id=tok_cfg.unk_id,
        bos_id=tok_cfg.bos_id,
        eos_id=tok_cfg.eos_id,
        pad_id=tok_cfg.pad_id,
        unk_piece=tok_cfg.unk_piece,
        bos_piece=tok_cfg.bos_piece,
        eos_piece=tok_cfg.eos_piece,
        pad_piece=tok_cfg.pad_piece,
        user_defined_symbols=tok_cfg.user_defined_symbols,
        # Llama-compatible behaviour / exact round-tripping.
        byte_fallback=True,
        split_digits=True,
        allow_whitespace_only_pieces=True,
        remove_extra_whitespaces=False,
        normalization_rule_name="identity",
        num_threads=8,
    )

    model_path = prefix.with_suffix(".model")
    if not model_path.exists():
        raise RuntimeError(f"SentencePiece training produced no model at {model_path}")
    return model_path


def load_sp(model_path: Path) -> spm.SentencePieceProcessor:
    """Load the fast encoder used for bulk tokenization."""
    sp = spm.SentencePieceProcessor()
    sp.Load(str(model_path))
    return sp


# ---------------------------------------------------------------------------
# HF wrapper (packaging only)
# ---------------------------------------------------------------------------

def build_hf_tokenizer(model_path: Path, out_dir: Path, chat_model: bool = False):
    """Wrap tokenizer.model as a transformers LlamaTokenizer and save it.

    ``chat_model=True`` sets the EOS token to ``<|im_end|>``. This matters at
    serving time: llama-server stops generating at the GGUF's EOS id, and for a
    ChatML model that has to be the turn terminator, not ``</s>``. The base
    text-completion model keeps ``</s>``.

    Saving emits ``tokenizer.model`` into ``out_dir``, which is what routes
    stage 8 down the SentencePiece path. Do not delete it.
    """
    from transformers import LlamaTokenizer

    out_dir.mkdir(parents=True, exist_ok=True)

    tok = LlamaTokenizer(
        vocab_file=str(model_path),
        unk_token=tok_cfg.unk_piece,
        bos_token=tok_cfg.bos_piece,
        eos_token=tok_cfg.im_end if chat_model else tok_cfg.eos_piece,
        pad_token=tok_cfg.pad_piece,
        add_bos_token=True,
        add_eos_token=False,
        legacy=False,
    )
    tok.chat_template = tok_cfg.chat_template
    tok.model_max_length = 512
    tok.save_pretrained(str(out_dir))

    # save_pretrained names the vocab file after the tokenizer class; make sure
    # a file literally called tokenizer.model exists, since that exact name is
    # what convert_hf_to_gguf.py probes for.
    canonical = out_dir / "tokenizer.model"
    if not canonical.exists():
        shutil.copy2(model_path, canonical)

    return tok


def assert_gguf_ready(out_dir: Path) -> None:
    """Fail loudly, now, if stage 8 would fall through to the hashed BPE path.

    Called at the end of stage 2 and again before conversion. Discovering this
    in stage 8 means re-running the export; discovering it here costs nothing.
    """
    model_file = out_dir / "tokenizer.model"
    if not model_file.exists():
        raise FileNotFoundError(
            f"{model_file} is missing.\n"
            "convert_hf_to_gguf.py would fall through to _set_vocab_gpt2(), which "
            "hashes the pre-tokenizer against a hardcoded registry and will raise "
            "'BPE pre-tokenizer was not recognized' for this custom vocabulary.\n"
            "Re-run build_hf_tokenizer() so tokenizer.model is written."
        )
    if model_file.stat().st_size < 1024:
        raise ValueError(f"{model_file} is suspiciously small ({model_file.stat().st_size} B)")


# ---------------------------------------------------------------------------
# Verification -- the stage 2 gate
# ---------------------------------------------------------------------------

def roundtrip_report(sp: spm.SentencePieceProcessor, texts: list[str]) -> dict:
    """Encode/decode every text and measure exactness plus compression.

    Round-tripping is only exact because of the normalization settings in
    ``train_sentencepiece``. If someone flips those back to defaults, this is
    the check that catches it.
    """
    failures: list[tuple[str, str]] = []
    n_chars = n_tokens = 0

    for t in texts:
        ids = sp.EncodeAsIds(t)
        back = sp.DecodeIds(ids)
        n_chars += len(t)
        n_tokens += len(ids)
        if back != t and len(failures) < 5:
            failures.append((t, back))

    return {
        "n_texts": len(texts),
        "exact": len(failures) == 0,
        "n_failures": len(failures),
        "examples": failures,
        "chars_per_token": (n_chars / n_tokens) if n_tokens else 0.0,
        "tokens_per_char": (n_tokens / n_chars) if n_chars else 0.0,
        "total_tokens": n_tokens,
    }


def vocab_stats(sp: spm.SentencePieceProcessor) -> dict:
    """Composition of the learned vocabulary."""
    pieces = [sp.IdToPiece(i) for i in range(sp.GetPieceSize())]
    byte_pieces = [p for p in pieces if p.startswith("<0x") and p.endswith(">")]
    return {
        "vocab_size": sp.GetPieceSize(),
        "byte_fallback_pieces": len(byte_pieces),
        "control_pieces": sum(1 for i in range(sp.GetPieceSize()) if sp.IsControl(i)),
        "unk_id": sp.unk_id(),
        "bos_id": sp.bos_id(),
        "eos_id": sp.eos_id(),
        "pad_id": sp.pad_id(),
        "im_start_id": sp.PieceToId(tok_cfg.im_start),
        "im_end_id": sp.PieceToId(tok_cfg.im_end),
        "longest_piece": max(pieces, key=len),
    }


def assert_special_tokens(sp: spm.SentencePieceProcessor) -> None:
    """The ChatML tokens must be single atomic ids, not subword sequences.

    If they split, the chat template silently stops working: the model sees
    fragments instead of a turn boundary, and llama-server never finds an EOS.
    """
    for piece in (tok_cfg.im_start, tok_cfg.im_end):
        pid = sp.PieceToId(piece)
        if pid == sp.unk_id():
            raise ValueError(f"{piece!r} is not in the vocabulary")

        # SentencePiece prepends a dummy whitespace prefix (add_dummy_prefix,
        # on by default), so encoding a bare control token yields [_, <piece>]
        # -- two ids -- even when the piece is perfectly atomic. Asserting
        # `ids == [pid]` therefore tests the dummy prefix, not atomicity.
        # Drop whitespace-only pieces before checking.
        ids = sp.EncodeAsIds(piece)
        core = [i for i in ids if sp.IdToPiece(i).replace("▁", "").strip() != ""]
        if core != [pid]:
            raise ValueError(
                f"{piece!r} tokenizes to {[sp.IdToPiece(i) for i in ids]} "
                f"(non-whitespace part {[sp.IdToPiece(i) for i in core]}) instead of "
                f"the single piece {piece!r}. It was not registered as a "
                "user_defined_symbol."
            )

        # The property that actually matters: the token stays a single id when
        # it appears mid-text, which is how the chat template uses it. A piece
        # can survive the bare-encode check and still be split in context.
        context = f"hello{piece}world"
        ctx_ids = sp.EncodeAsIds(context)
        if ctx_ids.count(pid) != 1:
            raise ValueError(
                f"{piece!r} does not survive as a single id inside text: "
                f"{context!r} -> {[sp.IdToPiece(i) for i in ctx_ids]}"
            )

    for name, expected in (
        ("unk", tok_cfg.unk_id), ("bos", tok_cfg.bos_id),
        ("eos", tok_cfg.eos_id), ("pad", tok_cfg.pad_id),
    ):
        actual = getattr(sp, f"{name}_id")()
        if actual != expected:
            raise ValueError(f"{name}_id is {actual}, expected {expected}")


def save_report(report: dict, path: Path) -> None:
    """Persist stage metrics next to the artifact they describe."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")


# ---------------------------------------------------------------------------
# Bulk encoding helper for stage 1/4
# ---------------------------------------------------------------------------

def encode_stream(
    sp: spm.SentencePieceProcessor,
    texts: Iterable[str],
    add_bos: bool = True,
    add_eos: bool = True,
    batch_size: int = 1000,
) -> Iterator[list[int]]:
    """Tokenize an iterable of documents, batched.

    BOS/EOS wrap each document so the model learns where stories start and end.
    Without EOS it never learns to stop, which shows up at inference as
    generation that runs until the token limit.
    """
    bos = [sp.bos_id()] if add_bos else []
    eos = [sp.eos_id()] if add_eos else []
    batch: list[str] = []

    for text in texts:
        batch.append(text)
        if len(batch) >= batch_size:
            for ids in sp.EncodeAsIds(batch):  # batched C++ call
                yield bos + ids + eos
            batch = []

    if batch:
        for ids in sp.EncodeAsIds(batch):
            yield bos + ids + eos
