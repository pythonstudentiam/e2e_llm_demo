# 04 — Pretraining

**Notebook:** `notebooks/colab/04_pretrain.ipynb`
**Code:** `src/tinyllm/train.py`

## What this stage does

Turns random weights into a model that writes English. ~5,000 steps, ~328M
tokens, ~45 minutes on a free Colab T4.

---

## fp16, not bf16 — and why that forces a GradScaler

A Colab T4 is **Turing (SM75)**, which has no bf16 support. That single hardware
fact drives the whole precision setup.

| | fp16 | bf16 |
|---|---|---|
| Exponent bits | 5 | 8 (same range as fp32) |
| Mantissa bits | 10 | 7 |
| Gradient underflow | **yes** | no |
| Needs loss scaling | **yes** | no |
| Hardware | Volta+ | Ampere+ |

fp16's narrow exponent range means small gradients simply **underflow to zero**
and training silently stalls — the loss plateaus and nothing looks obviously
broken.

Loss scaling fixes it: multiply the loss by a large factor before `.backward()`
so gradients land inside fp16's representable range, then divide back out before
the optimizer step. `GradScaler` does this and adapts the factor automatically,
backing off when it detects overflow.

**Occasional loss-scale warnings are expected**, not a failure — that's the scaler
searching for the right factor. On an A100 or newer you'd use bf16 and delete the
scaler entirely.

### Order of operations matters

```python
scaler.scale(loss).backward()
scaler.unscale_(optimizer)                    # MUST come before clipping
clip_grad_norm_(model.parameters(), 1.0)
scaler.step(optimizer)
scaler.update()
```

Clip before unscaling and the threshold is applied to gradients still multiplied
by the loss scale — numbers ~65,536× too large, so clipping never triggers. The
code runs, no error appears, and you simply have no gradient clipping. This is a
genuinely common bug.

---

## Gradient accumulation

The effective batch is 65,536 tokens, which won't fit in 16 GB at once, so it's
split into 4 micro-batches of 32×512.

Each micro-batch's loss is divided by `grad_accum_steps` **before** backward, so
the accumulated gradient equals the gradient of the mean loss over the full batch.
Forget the division and your effective learning rate is 4× what you configured.

## The learning-rate schedule

Linear warmup (200 steps) then cosine decay to 10% of peak.

**Warmup** exists because Adam's second-moment estimate is garbage for the first
few dozen steps. Stepping at full LR before it stabilizes is a reliable way to
blow up a run in its first minute.

**Cosine decay** spends most of the budget at a high LR and anneals smoothly at
the end. The final low-LR phase is where most of the perplexity improvement
actually lands.

## Weight decay on matrices only

```python
(decay if p.dim() >= 2 else no_decay).append(p)
```

Decay applies to weight matrices, not to RMSNorm gains or biases. Applying it to
normalization gains pulls them toward zero, which isn't regularization — it's
damage to a parameter whose whole job is to set scale.

---

## Durable checkpointing

**Free-tier Colab reclaims runtimes without warning. A checkpoint on the runtime's
local disk is not a checkpoint.** State is pushed to a Hub repo every 500 steps.

What gets saved, and why each piece:

| | why |
|---|---|
| model weights | obvious |
| optimizer state | Adam's two moments per parameter. Without them, resume restarts the moment estimates and the first few steps after resume are effectively unwarmed |
| scaler state | the current loss scale; rediscovering it wastes steps |
| **numpy RNG state** | the data sampler's position |
| torch/CUDA RNG state | dropout and sampling reproducibility |
| step counter | where the LR schedule is |

The RNG state is the one most tutorial code omits. Restore only the weights and
the data order silently restarts from the beginning — the model re-reads tokens it
has already seen while you believe it's making fresh progress. Nothing errors, the
loss curve looks plausible, and the run is quietly worse than it should be.

**Checkpoints are ~190 MB — roughly 3× the model** — because Adam stores two
moment tensors per parameter. That's why the push cadence is 500 steps, not 50.

## The smoke run

50 steps, ~1 minute, exercising every code path the real run uses: accumulation,
autocast, the scaler, unscale-then-clip, evaluation, sampling, checkpoint writing.

It calls the *same* `train()` function with a smaller step budget rather than a
simplified copy. A rehearsal that runs different code proves nothing.

Finding a bug here costs a minute. Finding it 40 minutes into the real run costs
40 minutes.

---

## What to watch

| signal | healthy | trouble |
|---|---|---|
| `loss` | starts ≈ 9.01 (`ln 8192`), falls fast to ~4, then grinds | flat, or NaN |
| `gn` (grad norm) | settles into a stable range | repeated spikes into the tens → LR too high |
| `mfu` | 15–30% | see below |
| val vs train | tracking together | val rising while train falls → overfitting |

**MFU (Model FLOPs Utilization)** is the fraction of the GPU's peak FLOPs you're
actually achieving. 15–30% is normal and *expected* for a model this small: the
matrices are too small to saturate the tensor cores, so kernel launch overhead and
memory bandwidth dominate. Large-model training runs target 40–55%. A low number
here is a property of the scale, not a bug to fix.

## Chinchilla and the token budget

[Hoffmann et al. (2022)](https://arxiv.org/abs/2203.15556) found that for a fixed
compute budget, models should be trained on roughly **20 tokens per parameter**.
We use 328M tokens for 15.7M parameters — 20.8 tokens/param, right in that range.

Worth knowing that Chinchilla optimizes *training* compute. If a model will be
served many times, it's rational to train well past the Chinchilla point to get a
smaller model at the same quality — which is why Llama 3 8B saw ~15T tokens,
roughly 1,800 tokens/param. Inference cost, not training cost, is what dominates
in deployment.

## Gate

- [x] Smoke run passed before the real run
- [x] Resume verified to restore optimizer moments, not just weights
- [x] Val loss well below `ln(V) = 9.01`
- [x] Checkpoints on the Hub
- [x] Output recognizably English

**Next:** [05 — Evaluation](05-eval.md)
