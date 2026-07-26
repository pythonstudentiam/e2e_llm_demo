"""Stage 5 -- evaluation.

Three things are measured, and it is worth being clear about what each is
actually good for.

**Perplexity** is the exponential of the mean cross-entropy: roughly, "how many
tokens is the model effectively choosing between at each step". It is the only
number here that is comparable across checkpoints of the same model, so it is
what we track. It is *not* comparable across models with different tokenizers --
a model with a smaller vocabulary gets a lower perplexity for free -- which is
why leaderboards stopped reporting it.

**Repetition metrics** catch the most common failure of small models: falling
into a loop. Perplexity barely moves when this happens, because each repeated
token is individually very predictable. This is the clearest case of a metric
looking fine while the output is unusable.

**Reading the samples** is the one that actually tells you whether it worked.
Automated metrics at this scale are a proxy; the standard benchmarks (MMLU,
HumanEval, and friends) would return pure noise on a 15.7M-parameter model
trained on children's stories, so they are deliberately not used.
"""

from __future__ import annotations

import json
import math
from collections import Counter
from pathlib import Path

import numpy as np
import torch

from tinyllm.config import data_cfg, gen_cfg, model_cfg
from tinyllm.data import iter_eval_batches


# ---------------------------------------------------------------------------
# Perplexity
# ---------------------------------------------------------------------------

@torch.no_grad()
def perplexity(model, tokens: np.ndarray, batch_size: int = 32, seq_len: int = data_cfg.seq_len,
               n_batches: int = 100, device: str = "cuda") -> dict:
    """Token-level perplexity over deterministic held-out windows.

    Losses are averaged in log space (i.e. mean cross-entropy, then exponentiate)
    rather than averaging the per-batch perplexities, which would be a different
    and wrong quantity.
    """
    model.eval()
    losses = []
    for x, y in iter_eval_batches(tokens, batch_size, seq_len, n_batches, device=device):
        with torch.autocast("cuda", dtype=torch.float16, enabled=device.startswith("cuda")):
            out = model(input_ids=x, labels=y)
        losses.append(out.loss.float().item())

    mean_loss = float(np.mean(losses))
    return {
        "loss": mean_loss,
        "perplexity": math.exp(min(mean_loss, 20)),
        "n_batches": n_batches,
        "n_tokens": n_batches * batch_size * seq_len,
        "std_loss": float(np.std(losses)),
    }


def bits_per_token(loss: float) -> float:
    """Cross-entropy in bits -- the compression view of what the model learned."""
    return loss / math.log(2)


# ---------------------------------------------------------------------------
# Generation quality
# ---------------------------------------------------------------------------

@torch.no_grad()
def generate(model, sp, prompt: str, max_new_tokens: int = gen_cfg.max_new_tokens,
             temperature: float = gen_cfg.temperature, top_p: float = gen_cfg.top_p,
             top_k: int = gen_cfg.top_k, repetition_penalty: float = 1.0,
             device: str = "cuda", seed: int | None = None) -> str:
    """Continue a prompt. Returns only the newly generated text."""
    if seed is not None:
        torch.manual_seed(seed)
    model.eval()

    ids = [sp.bos_id()] + sp.EncodeAsIds(prompt)
    x = torch.tensor([ids], dtype=torch.long, device=device)
    out = model.generate(
        x,
        max_new_tokens=max_new_tokens,
        do_sample=temperature > 0,
        temperature=max(temperature, 1e-5),
        top_p=top_p,
        top_k=top_k,
        repetition_penalty=repetition_penalty,
        pad_token_id=sp.pad_id(),
        eos_token_id=sp.eos_id(),
    )
    return sp.DecodeIds(out[0].tolist()[len(ids):])


def repetition_stats(text: str, max_n: int = 4) -> dict:
    """Distinct-n plus the worst repeated n-gram.

    ``distinct_n`` is unique n-grams over total n-grams: 1.0 means nothing ever
    repeats, low values mean the model is looping. Below ~0.6 for distinct-3,
    output usually reads as broken.
    """
    words = text.split()
    stats: dict = {"n_words": len(words)}
    if len(words) < max_n + 1:
        return stats

    for n in range(1, max_n + 1):
        grams = [tuple(words[i : i + n]) for i in range(len(words) - n + 1)]
        if not grams:
            continue
        counts = Counter(grams)
        stats[f"distinct_{n}"] = len(counts) / len(grams)
        if n == max_n:
            top, cnt = counts.most_common(1)[0]
            stats["most_repeated"] = " ".join(top)
            stats["most_repeated_count"] = cnt
    return stats


@torch.no_grad()
def sampling_sweep(model, sp, prompt: str | None = None, temperatures=(0.1, 0.5, 0.8, 1.0, 1.3),
                   device: str = "cuda", max_new_tokens: int = 120, seed: int = 0) -> list[dict]:
    """Generate at several temperatures from one prompt.

    The tradeoff is visible directly: low temperature is coherent and repetitive,
    high temperature is varied and incoherent. Where the balance sits is a
    property of the model, not a universal constant, which is why this is
    measured rather than assumed.
    """
    prompt = prompt or gen_cfg.eval_prompts[0]
    rows = []
    for t in temperatures:
        text = generate(model, sp, prompt, temperature=t, max_new_tokens=max_new_tokens,
                        device=device, seed=seed)
        row = {"temperature": t, "text": text}
        row.update(repetition_stats(text))
        rows.append(row)
    return rows


@torch.no_grad()
def sample_suite(model, sp, prompts=None, device: str = "cuda", seed: int = 0, **kw) -> list[dict]:
    """Generate from the fixed prompt set defined in config.

    Fixed prompts on purpose: the point is comparing the same inputs across the
    base model and the fine-tuned model in stage 6.
    """
    prompts = prompts or gen_cfg.eval_prompts
    out = []
    for p in prompts:
        text = generate(model, sp, p, device=device, seed=seed, **kw)
        row = {"prompt": p, "generated": text}
        row.update(repetition_stats(text))
        out.append(row)
    return out


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def full_report(model, sp, val_tokens, device: str = "cuda", n_batches: int = 100) -> dict:
    """Everything stage 5 measures, in one dict. Saved with the model card."""
    ppl = perplexity(model, val_tokens, n_batches=n_batches, device=device)
    report = {
        "n_params": sum(p.numel() for p in model.parameters()),
        "val_loss": ppl["loss"],
        "val_perplexity": ppl["perplexity"],
        "bits_per_token": bits_per_token(ppl["loss"]),
        "eval_tokens": ppl["n_tokens"],
        "samples": sample_suite(model, sp, device=device),
        "temperature_sweep": sampling_sweep(model, sp, device=device),
    }
    return report


def print_report(report: dict) -> None:
    print(f"  params           {report['n_params']:,}")
    print(f"  val loss         {report['val_loss']:.4f}")
    print(f"  val perplexity   {report['val_perplexity']:.2f}")
    print(f"  bits/token       {report['bits_per_token']:.3f}")
    print(f"  (random baseline would be {math.log(model_cfg.vocab_size):.2f} loss / "
          f"{model_cfg.vocab_size} perplexity)")
    print("\n  samples:")
    for s in report["samples"]:
        d3 = s.get("distinct_3")
        d3s = f"{d3:.2f}" if d3 is not None else "n/a"
        print(f"\n    prompt:  {s['prompt']}")
        print(f"    output:  {s['generated'][:280]}")
        print(f"    distinct-3: {d3s}")


def save_report(report: dict, path: Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")


def compare(before: dict, after: dict, label_a: str = "base", label_b: str = "sft") -> None:
    """Side-by-side for the stage 6 before/after.

    Note that SFT usually makes *perplexity on the pretraining distribution
    slightly worse* while making the model far more useful. That is expected,
    not a regression -- it is the clearest illustration in this project of why a
    single metric should not be the target.
    """
    print(f"{'metric':<20} {label_a:>12} {label_b:>12}")
    print("-" * 46)
    for k in ("val_loss", "val_perplexity", "bits_per_token"):
        if k in before and k in after:
            print(f"{k:<20} {before[k]:>12.4f} {after[k]:>12.4f}")
