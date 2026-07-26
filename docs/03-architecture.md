# 03 — Architecture

**Notebook:** `notebooks/colab/03_architecture.ipynb`
**Code:** `src/tinyllm/model_scratch.py`, `src/tinyllm/parity.py`

## What this stage does

Implements the Llama decoder from scratch, then **proves** it computes the same
function as `transformers.LlamaForCausalLM`.

## Why two implementations

| | from scratch | HuggingFace |
|---|---|---|
| Purpose | understanding | what actually trains |
| Read it | yes — ~200 lines | no |
| Converts to GGUF | you'd hand-map every tensor | works out of the box |

Training HF's version means stage 8 carries no conversion risk. Writing our own
means you understand what's inside it. The equality proof is what makes the
second claim trustworthy rather than aspirational.

**An *almost* correct transformer is the worst outcome.** It trains, the loss
falls, and the bug surfaces as "the model just isn't very good" after the GPU time
is spent. RoPE convention errors and GQA head-mapping errors both fail exactly
that way.

---

## The four ingredients

### RMSNorm

LayerNorm without mean subtraction:

```
y = x / sqrt(mean(x²) + ε) · g
```

Cheaper than LayerNorm (no mean pass, no bias) and empirically as good.
[Zhang & Sennrich (2019)](https://arxiv.org/abs/1910.07467).

**The subtlety is dtype.** The variance of an fp16 activation can lose all its
precision or overflow outright, so normalization is computed in fp32 and cast
back. Getting this wrong produces silent divergence during mixed-precision
training — the loss looks fine for a while and then goes to NaN.

### RoPE — rotary position embeddings

Position is encoded by *rotating* q and k in 2D subspaces rather than adding a
position vector to the residual stream.
[Su et al. (2021)](https://arxiv.org/abs/2104.09864).

Frequencies are geometrically spaced, `1 / θ^(2i/d)`. Low-index pairs rotate fast
(fine local position), high-index pairs rotate slowly (coarse global position).
Because a rotation by *m* composed with a rotation by *−n* is a rotation by
*m−n*, attention scores end up depending only on **relative** distance — with no
explicit relative-position bias term anywhere.

**The convention trap.** HuggingFace pairs dimension *i* with *i + d/2*:

```python
def rotate_half(x):
    x1, x2 = x[..., :d//2], x[..., d//2:]
    return torch.cat((-x2, x1), dim=-1)
```

The RoPE paper's notation suggests pairing *i* with *i+1*. The two are equivalent
up to a permutation of the head dimension, but **not interchangeable for a given
set of weights**. Matching HF here is precisely what makes the parity check pass —
and a mismatch is one of those bugs that trains happily to a mediocre result.

### GQA — grouped-query attention

6 query heads share 2 key/value heads.
[Ainslie et al. (2023)](https://arxiv.org/abs/2305.13245).

`repeat_kv` is a view-and-reshape, so it costs **no FLOPs**. The saving is in the
**cache**: only 2 heads' worth of K and V are ever stored, cutting KV cache memory
by 67%.

That matters because the KV cache, not the weights, is what limits long-context
inference. At 512 tokens here it's a rounding error; at 128k tokens on a 70B model
it's the difference between fitting on a GPU and not. Using GQA at this scale is
partly pedagogical — `num_key_value_heads=6` would have been simpler and equally
good — but it makes the mechanism concrete and exercisable.

### SwiGLU

```
down(silu(gate(x)) · up(x))
```

A gated MLP: `gate` decides, per channel, how much of `up` survives.
[Shazeer (2020)](https://arxiv.org/abs/2002.05202).

Three matrices instead of two, so the intermediate width is ~2.67× hidden rather
than 4× for the same parameter count. The MLP still holds ~60% of the model's
parameters, which is why `intermediate_size` is the first knob to reach for when
resizing.

---

## Other decisions

**Pre-norm.** Normalize *going into* each sublayer rather than after it, so the
residual stream is never renormalized and gradients reach layer 0 undiminished.
This is why deep transformers train without the warmup tricks post-norm needed.

**No biases anywhere.** Llama drops them from every linear layer. With
normalization present they cost parameters and buy nothing.

**Tied embeddings.** `lm_head.weight is embed_tokens.weight` — one tensor, two
uses. Saves 3.1M parameters, 20% of the model. Sound at this scale because both
are maps between the same two spaces (token ids ↔ hidden states). Large models
often untie them, since 100M+ embedding parameters are affordable and the two
roles do benefit from specializing.

---

## The parity checks

| check | catches |
|---|---|
| parameter counts | shape errors, and disagreement with `config.param_breakdown()` |
| logit parity (<1e-4) | RoPE convention, GQA head order, RMSNorm dtype |
| loss parity | label-shift convention mismatch |
| **cached == uncached generation** | **RoPE position offset bugs** |

The fourth earns its keep. Logit and loss parity both run a single forward pass
starting at position 0, so they never exercise a non-zero RoPE offset. A KV-cache
bug — stale positions, re-rotating cached keys, an off-by-one in the offset —
leaves **training completely unaffected** and only corrupts inference. It survives
every other test here, and would surface much later as "the GGUF is worse than the
checkpoint", pointing suspicion at entirely the wrong stage.

HF is forced to `attn_implementation="eager"` for these checks. Its default (SDPA)
is numerically fine but fuses operations differently, producing ~1e-5 drift —
enough to make a strict tolerance flaky for reasons unrelated to correctness.

## Gate

- [x] Parameter counts agree across both models and `config.py`
- [x] Logits match to < 1e-4
- [x] Loss matches
- [x] Cached generation == uncached generation
- [x] Untrained loss ≈ `ln(vocab_size)` = 9.01

**Next:** [04 — Pretraining](04-pretraining.md)
