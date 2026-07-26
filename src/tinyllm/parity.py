"""Stage 3 gate -- prove the from-scratch model equals HuggingFace's.

A from-scratch transformer that is *almost* right is worse than useless: it
trains, the loss goes down, and the bug shows up as "the model just isn't very
good" after you have already spent the GPU time. RoPE convention errors and GQA
head-mapping errors both fail exactly that way.

So rather than trusting the implementation, we load one set of weights into both
models and check they compute the same function. Four checks:

  1. parameter counts agree with each other and with config.param_breakdown()
  2. logits match on random input
  3. loss matches
  4. cached generation matches uncached generation

Check 4 is the one that catches RoPE offset bugs, which are invisible in checks
2 and 3 because those never exercise a non-zero position offset.
"""

from __future__ import annotations

import torch

from tinyllm.config import ModelConfig, model_cfg
from tinyllm.model_scratch import TinyLlamaForCausalLM

LOGIT_TOL = 1e-4
LOSS_TOL = 1e-4


def build_pair(cfg: ModelConfig = model_cfg, seed: int = 0, device: str = "cpu"):
    """Build both models on identical weights.

    HF is forced to ``attn_implementation='eager'``. Its default (SDPA) is
    numerically fine but fuses operations differently, which shows up as ~1e-5
    logit drift -- enough to make a strict tolerance flaky for reasons that have
    nothing to do with correctness.
    """
    from transformers import LlamaForCausalLM

    torch.manual_seed(seed)
    hf_cfg = cfg.to_hf_config()
    hf_cfg._attn_implementation = "eager"
    hf = LlamaForCausalLM(hf_cfg).to(device).eval()

    scratch = TinyLlamaForCausalLM(cfg).to(device).eval()

    src = hf.state_dict()
    missing, unexpected = scratch.load_state_dict(src, strict=False)

    # Tied models may omit lm_head.weight; rotary tables are non-persistent
    # buffers and legitimately absent from both. Anything else is a real
    # name mismatch and means the two models are not comparable.
    allowed = {"lm_head.weight"}
    real_missing = [k for k in missing if k not in allowed and "rotary" not in k]
    real_unexpected = [k for k in unexpected if "rotary" not in k]

    if real_missing or real_unexpected:
        raise RuntimeError(
            "State dict does not transfer cleanly between the two models.\n"
            f"  missing in scratch:   {real_missing}\n"
            f"  unexpected from HF:   {real_unexpected}\n"
            "Module names must match HuggingFace's exactly."
        )

    if "lm_head.weight" in missing:
        if not cfg.tie_word_embeddings:
            raise RuntimeError("lm_head.weight missing but embeddings are not tied")
        # Tying is re-established by identity, so nothing to copy.
        assert scratch.lm_head.weight is scratch.model.embed_tokens.weight

    return hf, scratch


def check_param_counts(hf, scratch, cfg: ModelConfig = model_cfg) -> dict:
    """Both models, and the arithmetic in config.py, must agree."""
    hf_n = sum(p.numel() for p in hf.parameters())
    sc_n = sum(p.numel() for p in scratch.parameters())
    predicted = cfg.param_breakdown()["total"]

    if hf_n != sc_n:
        raise AssertionError(f"parameter count differs: HF={hf_n:,} scratch={sc_n:,}")
    if sc_n != predicted:
        raise AssertionError(
            f"config.param_breakdown() predicts {predicted:,} but the model has {sc_n:,}. "
            "One of them is wrong."
        )
    return {"hf": hf_n, "scratch": sc_n, "predicted": predicted}


def check_logits(hf, scratch, cfg: ModelConfig = model_cfg, batch: int = 2, seq: int = 64) -> dict:
    """Forward both models on the same random tokens."""
    torch.manual_seed(1234)
    ids = torch.randint(0, cfg.vocab_size, (batch, seq))

    with torch.no_grad():
        hf_logits = hf(input_ids=ids).logits
        sc_logits, _, _ = scratch(ids)

    diff = (hf_logits - sc_logits).abs()
    max_diff = diff.max().item()
    if max_diff > LOGIT_TOL:
        raise AssertionError(
            f"logits diverge: max|delta| = {max_diff:.3e} > {LOGIT_TOL:.0e}.\n"
            "Most likely causes: RoPE half-rotation convention, GQA head "
            "repetition order, or RMSNorm computed in the wrong dtype."
        )
    return {"max_abs_diff": max_diff, "mean_abs_diff": diff.mean().item(), "shape": tuple(hf_logits.shape)}


def check_loss(hf, scratch, cfg: ModelConfig = model_cfg, batch: int = 2, seq: int = 64) -> dict:
    """Both models must produce the same loss on the same labels.

    HF shifts labels internally; so does our forward. If only one did, the loss
    would differ by roughly one token of context and this catches it.
    """
    torch.manual_seed(4321)
    ids = torch.randint(0, cfg.vocab_size, (batch, seq))

    with torch.no_grad():
        hf_loss = hf(input_ids=ids, labels=ids).loss.item()
        _, sc_loss, _ = scratch(ids, labels=ids)
        sc_loss = sc_loss.item()

    delta = abs(hf_loss - sc_loss)
    if delta > LOSS_TOL:
        raise AssertionError(
            f"loss differs: HF={hf_loss:.6f} scratch={sc_loss:.6f} (delta {delta:.2e}). "
            "Check the label-shift convention."
        )
    # A randomly initialised model should sit near ln(vocab_size).
    import math
    expected = math.log(cfg.vocab_size)
    return {"hf": hf_loss, "scratch": sc_loss, "delta": delta, "ln_vocab": expected}


def check_kv_cache(scratch, cfg: ModelConfig = model_cfg, prompt_len: int = 8, seq_len: int = 24) -> dict:
    """Incremental cached decoding must reproduce a single full-sequence forward.

    This is the check that matters most. A KV cache bug -- stale positions,
    rotating cached keys twice, an off-by-one in the offset, or simply never
    writing the cache -- leaves training completely unaffected and only corrupts
    inference, so it survives every other test in this file.

    Logits are compared rather than sampled tokens. Comparing greedy token ids
    looks stricter but is actually brittle: an untrained model's logits are
    nearly tied, so a 1e-7 difference can flip an argmax and the two sequences
    then diverge forever for reasons that have nothing to do with correctness.
    Comparing logits tests the real invariant and gives a magnitude, not a
    yes/no.
    """
    torch.manual_seed(7)
    seq = torch.randint(0, cfg.vocab_size, (1, seq_len))

    # Reference: the whole sequence in one forward, no cache.
    with torch.no_grad():
        full_logits, _, _ = scratch(seq)

    # Incremental: prefill the prompt, then feed one token at a time.
    caches = [None] * cfg.num_hidden_layers
    with torch.no_grad():
        lg, _, caches = scratch(seq[:, :prompt_len], kv_caches=caches, offset=0)

    # The most direct possible check, and the one that catches a cache that is
    # silently never written.
    if caches is None or caches[0] is None:
        raise AssertionError(
            "The KV cache was not populated by the prefill pass. Every "
            "subsequent token would attend only to itself."
        )
    past_len = caches[0][0].shape[2]
    if past_len != prompt_len:
        raise AssertionError(f"cache holds {past_len} positions after prefilling {prompt_len}")

    incremental = [lg[:, -1]]
    for t in range(prompt_len, seq_len):
        with torch.no_grad():
            lg, _, caches = scratch(seq[:, t : t + 1], kv_caches=caches, offset=t)
        incremental.append(lg[:, -1])

        expected = t + 1
        got = caches[0][0].shape[2]
        if got != expected:
            raise AssertionError(f"after step {t} the cache holds {got} positions, expected {expected}")

    # incremental[j] is the prediction following position prompt_len-1+j.
    max_diff = 0.0
    for j, inc in enumerate(incremental):
        ref = full_logits[:, prompt_len - 1 + j, :]
        max_diff = max(max_diff, (ref - inc).abs().max().item())

    if max_diff > LOGIT_TOL:
        raise AssertionError(
            f"cached decoding diverges from a full forward: max|delta| = {max_diff:.3e}.\n"
            "Likely a RoPE position offset bug, or cached keys being re-rotated."
        )

    # Informational: with matching logits, greedy tokens should agree too --
    # but near-ties in an untrained model can still flip one, so this is
    # reported rather than asserted.
    with torch.no_grad():
        cached_gen = scratch.generate(seq[:, :prompt_len], max_new_tokens=8, temperature=0.0, use_cache=True)
        uncached_gen = scratch.generate(seq[:, :prompt_len], max_new_tokens=8, temperature=0.0, use_cache=False)

    return {
        "positions_checked": len(incremental),
        "max_abs_diff": max_diff,
        "final_cache_len": caches[0][0].shape[2],
        "greedy_identical": torch.equal(cached_gen, uncached_gen),
    }


def run_all(cfg: ModelConfig = model_cfg, verbose: bool = True) -> dict:
    """Run every check. Raises on the first failure; returns a report if clean."""
    results: dict = {}

    if verbose:
        print("Building both models on identical weights...")
    hf, scratch = build_pair(cfg)

    steps = [
        ("parameter counts", lambda: check_param_counts(hf, scratch, cfg)),
        ("logit parity", lambda: check_logits(hf, scratch, cfg)),
        ("loss parity", lambda: check_loss(hf, scratch, cfg)),
        ("kv-cache parity", lambda: check_kv_cache(scratch, cfg)),
    ]

    for name, fn in steps:
        res = fn()
        results[name.replace(" ", "_")] = res
        if verbose:
            print(f"  PASS  {name:<20} {_fmt(res)}")

    if verbose:
        print("\nThe from-scratch implementation is a verified reference for what gets trained.")
    return results


def _fmt(res: dict) -> str:
    if "max_abs_diff" in res:
        return f"max|delta|={res['max_abs_diff']:.2e}"
    if "delta" in res:
        return f"HF={res['hf']:.4f} scratch={res['scratch']:.4f} (ln V={res['ln_vocab']:.4f})"
    if "predicted" in res:
        return f"{res['scratch']:,} params, matches config.py"
    if "positions_checked" in res:
        greedy = "greedy identical" if res["greedy_identical"] else "greedy differs (near-tie)"
        return (f"{res['positions_checked']} positions, max|delta|={res['max_abs_diff']:.2e}, "
                f"cache={res['final_cache_len']}, {greedy}")
    return ""


if __name__ == "__main__":
    run_all()
