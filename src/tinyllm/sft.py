"""Stage 6 -- instruction tuning.

Pretraining produced a text completer: give it "Once upon a time" and it keeps
writing. This stage turns it into something that answers a request. Two ideas do
all the work.

**Chat template.** Turns are wrapped in ChatML so the model can tell "what I was
asked" apart from "what I should say". The same template is written into
tokenizer_config.json, from there into the GGUF metadata, and from there
llama-server uses it to format incoming API requests. One string, four hops --
if it disagrees anywhere along that chain, the served model sees prompts in a
format it was never trained on and produces subtly worse output for no visible
reason.

**Completion-only loss masking.** Prompt tokens get label ``-100`` so they
contribute no gradient. Without this the model spends capacity learning to
generate instructions, which is not the job. This fails *silently* -- training
still converges, just to a worse model -- so notebook 06 decodes a real batch and
asserts the mask sits exactly where it should.

On the dataset format
---------------------
TinyStoriesInstruct is stored one *line* per row, not one example per row.
Records are separated by a ``<|endoftext|>`` line, and the header fields appear
in arbitrary order::

    Features: Dialogue
    Words: quit, oak, gloomy
    Summary: Sara and Ben were playing in the park...
    Story:
    <blank>
    Sara and Ben were playing in the park. ...
    <|endoftext|>
    Summary: Lily steals a new bike...      <- different field order
    Words: ride, work, upset
    Features: Dialogue, BadEnding, MoralValue
    Story:
    ...

So the reader below accumulates lines into records rather than treating rows as
examples, and matches headers by prefix rather than by position.
"""

from __future__ import annotations

import json
import math
import time
from pathlib import Path
from typing import Iterable, Iterator

import numpy as np

from tinyllm.config import SFTConfig, gen_cfg, sft_cfg, tok_cfg

# torch is imported lazily inside collate/train_sft/chat rather than here, so the
# record parser and instruction builder can be imported -- and tested -- on the
# local tier, which has no PyTorch. The parsing is pure data manipulation and is
# where the format bugs live, so it is worth being able to exercise it cheaply.

RECORD_SEP = "<|endoftext|>"
STORY_HEADER = "Story:"
FIELD_PREFIXES = ("Features:", "Words:", "Summary:")


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------

def iter_instruct_records(lines: Iterable[str], limit: int | None = None) -> Iterator[dict]:
    """Reassemble line-per-row data into records.

    Yields dicts with any of ``features``/``words``/``summary`` that were
    present, plus ``story``. Records missing a story are skipped rather than
    yielded half-formed.
    """
    buf: list[str] = []
    n = 0

    def finish(block: list[str]) -> dict | None:
        rec: dict = {}
        story_lines: list[str] = []
        in_story = False
        for raw in block:
            line = raw.rstrip("\n")
            if not in_story:
                if line.startswith(STORY_HEADER):
                    in_story = True
                    tail = line[len(STORY_HEADER):].strip()
                    if tail:
                        story_lines.append(tail)
                    continue
                for pref in FIELD_PREFIXES:
                    if line.startswith(pref):
                        rec[pref[:-1].lower()] = line[len(pref):].strip()
                        break
            else:
                story_lines.append(line)
        story = "\n".join(story_lines).strip()
        if not story:
            return None
        rec["story"] = story
        return rec

    for raw in lines:
        line = (raw or "").rstrip("\n")
        if line.strip() == RECORD_SEP:
            rec = finish(buf)
            buf = []
            if rec:
                n += 1
                yield rec
                if limit is not None and n >= limit:
                    return
        else:
            buf.append(line)

    if buf:
        rec = finish(buf)
        if rec:
            yield rec


def stream_instruct_records(dataset_id: str = "roneneldan/TinyStoriesInstruct",
                            split: str = "train", limit: int | None = None) -> Iterator[dict]:
    """Stream and parse the instruct dataset straight from the Hub."""
    from datasets import load_dataset

    ds = load_dataset(dataset_id, split=split, streaming=True)
    yield from iter_instruct_records((row["text"] for row in ds), limit=limit)


# ---------------------------------------------------------------------------
# Instruction synthesis
# ---------------------------------------------------------------------------

_WORD_TEMPLATES = (
    "Write a story using the words: {words}.",
    "Write a short story that includes these words: {words}.",
    "Tell me a story with the words {words} in it.",
)
_SUMMARY_TEMPLATES = (
    "Write a story about this: {summary}",
    "Here is what should happen in the story: {summary} Write it.",
)
_COMBINED_TEMPLATES = (
    "Write a story using the words: {words}. The story should be about: {summary}",
    "Write a short story about: {summary} Include the words {words}.",
)
_FEATURE_HINTS = {
    "dialogue": "Include some dialogue.",
    "moralvalue": "The story should teach a moral lesson.",
    "badending": "The story should have a sad ending.",
    "foreshadowing": "Use some foreshadowing.",
    "twist": "Include a twist.",
    "conflict": "Include a conflict.",
}


def build_instruction(rec: dict, rng: np.random.Generator) -> str:
    """Turn a record's metadata into a natural-language request.

    Templates are sampled rather than fixed so the model learns the *task*
    rather than one exact phrasing. A single template produces a model that
    breaks the moment a user words the request differently.
    """
    words = rec.get("words", "").strip()
    summary = rec.get("summary", "").strip()
    features = rec.get("features", "").strip()

    if words and summary:
        tpl = _COMBINED_TEMPLATES[rng.integers(len(_COMBINED_TEMPLATES))]
        instr = tpl.format(words=words, summary=summary)
    elif words:
        tpl = _WORD_TEMPLATES[rng.integers(len(_WORD_TEMPLATES))]
        instr = tpl.format(words=words)
    elif summary:
        tpl = _SUMMARY_TEMPLATES[rng.integers(len(_SUMMARY_TEMPLATES))]
        instr = tpl.format(summary=summary)
    else:
        instr = "Write a short story."

    # Mention a feature only sometimes, so the model doesn't come to expect it.
    if features and rng.random() < 0.5:
        for f in features.split(","):
            hint = _FEATURE_HINTS.get(f.strip().lower())
            if hint:
                instr += " " + hint
                break

    return instr


# ---------------------------------------------------------------------------
# Encoding with completion-only masking
# ---------------------------------------------------------------------------

def chatml_prompt(instruction: str) -> str:
    """Everything up to where the assistant starts speaking."""
    return (
        f"{tok_cfg.im_start}user\n{instruction}{tok_cfg.im_end}\n"
        f"{tok_cfg.im_start}assistant\n"
    )


def chatml_completion(story: str) -> str:
    return f"{story}{tok_cfg.im_end}\n"


def encode_example(sp, instruction: str, story: str, seq_len: int = sft_cfg.seq_len,
                   ignore_index: int = sft_cfg.ignore_index) -> dict | None:
    """Encode one example and mask the prompt out of the loss.

    Prompt and completion are encoded *separately* and concatenated, rather than
    encoding the joined string and slicing. Slicing would be wrong: BPE can merge
    across the boundary, so the prompt's token count in the joined string need
    not match its count alone, and the mask would land in the wrong place.
    """
    prompt_ids = [sp.bos_id()] + sp.EncodeAsIds(chatml_prompt(instruction))
    completion_ids = sp.EncodeAsIds(chatml_completion(story))

    input_ids = prompt_ids + completion_ids
    labels = [ignore_index] * len(prompt_ids) + list(completion_ids)

    if len(input_ids) > seq_len:
        # Truncating would cut the story off mid-sentence and teach the model to
        # stop arbitrarily. Dropping the example is cheaper than that damage.
        return None
    if not completion_ids:
        return None

    return {"input_ids": input_ids, "labels": labels, "n_prompt": len(prompt_ids)}


def build_sft_dataset(sp, limit: int = 60_000, seq_len: int = sft_cfg.seq_len,
                      seed: int = sft_cfg.seed, split: str = "train") -> list[dict]:
    """Materialize the SFT set. Small enough to hold in memory."""
    rng = np.random.default_rng(seed)
    out: list[dict] = []
    n_seen = n_dropped = 0

    for rec in stream_instruct_records(split=split, limit=limit):
        n_seen += 1
        ex = encode_example(sp, build_instruction(rec, rng), rec["story"], seq_len=seq_len)
        if ex is None:
            n_dropped += 1
            continue
        out.append(ex)

    print(f"  {len(out):,} examples kept, {n_dropped:,} dropped as too long "
          f"({n_dropped / max(1, n_seen):.1%} of {n_seen:,})")
    return out


def collate(batch: list[dict], pad_id: int, ignore_index: int = sft_cfg.ignore_index, device: str = "cuda"):
    """Right-pad to the longest example in the batch.

    Padding is masked in ``attention_mask`` *and* in the labels. Masking only one
    leaves the model either attending to garbage or training on it.
    """
    import torch

    maxlen = max(len(b["input_ids"]) for b in batch)
    input_ids, labels, attn = [], [], []

    for b in batch:
        pad = maxlen - len(b["input_ids"])
        input_ids.append(b["input_ids"] + [pad_id] * pad)
        labels.append(b["labels"] + [ignore_index] * pad)
        attn.append([1] * len(b["input_ids"]) + [0] * pad)

    return (
        torch.tensor(input_ids, dtype=torch.long, device=device),
        torch.tensor(labels, dtype=torch.long, device=device),
        torch.tensor(attn, dtype=torch.long, device=device),
    )


# ---------------------------------------------------------------------------
# The stage 6 gate
# ---------------------------------------------------------------------------

def describe_masking(sp, example: dict, ignore_index: int = sft_cfg.ignore_index) -> dict:
    """Decode an example and show exactly what the loss does and does not see.

    Print this. The failure it catches -- a mask that is off, or absent -- costs
    nothing to see here and is invisible everywhere else.
    """
    ids, labels = example["input_ids"], example["labels"]
    masked = [i for i, lab in zip(ids, labels) if lab == ignore_index]
    supervised = [i for i, lab in zip(ids, labels) if lab != ignore_index]

    report = {
        "n_tokens": len(ids),
        "n_masked": len(masked),
        "n_supervised": len(supervised),
        "masked_text": sp.DecodeIds(masked),
        "supervised_text": sp.DecodeIds(supervised),
    }

    if len(masked) != example["n_prompt"]:
        raise AssertionError(
            f"{len(masked)} tokens masked but the prompt is {example['n_prompt']} tokens."
        )
    if any(lab != ignore_index for lab in labels[: example["n_prompt"]]):
        raise AssertionError("A prompt token is contributing to the loss.")
    if all(lab == ignore_index for lab in labels):
        raise AssertionError("Every token is masked -- this example would produce no gradient.")

    return report


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_sft(model, sp, dataset: list[dict], cfg: SFTConfig = sft_cfg,
              out_dir: Path = Path("checkpoints/sft"), device: str = "cuda",
              max_steps: int | None = None):
    """Fine-tune the pretrained model on instructions.

    The learning rate is ~6x below pretraining. Instruction tuning is meant to
    reshape output format, not relearn language; too high an LR here erases the
    pretrained knowledge and the model gets worse at the thing it was good at.
    """
    import torch

    from tinyllm.train import make_optimizer

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    total_steps = max_steps or cfg.max_steps

    torch.manual_seed(cfg.seed)
    rng = np.random.default_rng(cfg.seed)

    optimizer, _ = make_optimizer(model, cfg)
    scaler = torch.amp.GradScaler("cuda", enabled=device.startswith("cuda"))
    model.train()

    history: list[dict] = []
    t0 = time.time()
    pad_id = sp.pad_id()

    def lr_at(step: int) -> float:
        if step < cfg.warmup_steps:
            return cfg.learning_rate * (step + 1) / cfg.warmup_steps
        prog = (step - cfg.warmup_steps) / max(1, total_steps - cfg.warmup_steps)
        floor = cfg.learning_rate * cfg.min_lr_ratio
        return floor + 0.5 * (1 + math.cos(math.pi * min(prog, 1.0))) * (cfg.learning_rate - floor)

    print(f"  {len(dataset):,} examples | {total_steps:,} steps x "
          f"{cfg.micro_batch_size * cfg.grad_accum_steps} examples/step")

    for step in range(total_steps):
        lr = lr_at(step)
        for g in optimizer.param_groups:
            g["lr"] = lr

        optimizer.zero_grad(set_to_none=True)
        accum = 0.0
        for _ in range(cfg.grad_accum_steps):
            idx = rng.integers(0, len(dataset), size=cfg.micro_batch_size)
            x, y, attn = collate([dataset[i] for i in idx], pad_id, cfg.ignore_index, device)
            with torch.autocast("cuda", dtype=torch.float16, enabled=device.startswith("cuda")):
                out = model(input_ids=x, attention_mask=attn, labels=y)
                loss = out.loss / cfg.grad_accum_steps
            scaler.scale(loss).backward()
            accum += loss.item()

        scaler.unscale_(optimizer)
        gn = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip).item()
        scaler.step(optimizer)
        scaler.update()

        if (step + 1) % cfg.log_every == 0 or step == 0:
            history.append({"step": step + 1, "loss": accum, "lr": lr, "grad_norm": gn})
            print(f"  step {step + 1:>5,}/{total_steps:,} | loss {accum:.4f} | lr {lr:.2e} | gn {gn:5.2f}")

        if (step + 1) % cfg.checkpoint_every == 0 or (step + 1) == total_steps:
            torch.save({"step": step + 1, "model": model.state_dict()}, out_dir / "sft_latest.pt")

    print(f"\n  done in {(time.time() - t0) / 60:.1f} min")
    (out_dir / "sft_history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    return model, history


def chat(model, sp, instruction: str, max_new_tokens: int = gen_cfg.max_new_tokens,
         temperature: float = gen_cfg.temperature, device: str = "cuda") -> str:
    """Single-turn generation through the chat template.

    Stops at ``<|im_end|>`` -- the same token llama-server will stop at once the
    template reaches the GGUF, which is why the tokenizer for the chat model
    reports it as EOS.
    """
    import torch

    model.eval()
    ids = [sp.bos_id()] + sp.EncodeAsIds(chatml_prompt(instruction))
    x = torch.tensor([ids], dtype=torch.long, device=device)
    im_end = sp.PieceToId(tok_cfg.im_end)

    with torch.no_grad():
        out = model.generate(
            x,
            max_new_tokens=max_new_tokens,
            do_sample=temperature > 0,
            temperature=max(temperature, 1e-5),
            top_p=gen_cfg.top_p,
            top_k=gen_cfg.top_k,
            repetition_penalty=gen_cfg.repetition_penalty,
            pad_token_id=sp.pad_id(),
            eos_token_id=im_end,
        )
    model.train()
    generated = out[0].tolist()[len(ids):]
    text = sp.DecodeIds(generated)
    return text.replace(tok_cfg.im_end, "").strip()
