# 00 — Overview: the shape of the lifecycle

This document is the conceptual map. Each stage has its own doc with the details; this one explains *why the stages exist at all* and how they connect.

---

## The one-sentence version

A language model is a function that predicts the next token; everything in this pipeline is either **deciding what a token is** (stage 2), **deciding how the function is shaped** (stage 3), **fitting it to data** (stages 1, 4, 6), **checking whether it worked** (stage 5), or **getting it into a form something else can run** (stages 7–10).

Most of the engineering is in that last category, which is the part tutorials usually skip.

---

## Why the stages are in this order

### Data comes before the tokenizer

The tokenizer is *fit* to the corpus — it learns which byte sequences are frequent enough to deserve their own id. Train it on the wrong distribution and every downstream stage pays a permanent tax in wasted context.

### The tokenizer comes before the model

`vocab_size` is a model dimension. At our scale it's the *dominant* one: an 8192-vocab embedding table is 3.1M parameters — 20% of the whole model. Had we reused Llama-3's 128k vocab, the embedding table alone would be 49M parameters, three times larger than everything else combined, and the model would spend its capacity memorizing tokens it never sees.

This is why "just use an off-the-shelf tokenizer" is bad advice at small scale, and fine advice at large scale.

### Architecture comes before training, and gets *proved* first

Stage 3 writes the transformer twice — once from scratch to be read, once as HuggingFace's `LlamaForCausalLM` to be trained — and asserts they produce identical logits. This is not busywork:

- A silent architecture bug costs you a 45-minute training run to discover.
- The from-scratch version is how you *understand* RoPE and GQA; the HF version is what `convert_hf_to_gguf.py` knows how to read.
- The equality assertion is what lets you trust the from-scratch one as a reference.

### Pretraining before instruction tuning

Pretraining teaches the model *language*. Instruction tuning teaches it *what to do with a prompt*. The second is cheap (~1.5k steps) and only works because the first was expensive (~5k steps at 4x the batch size).

Do them in the other order and you have nothing to align.

### Evaluation between them, and again at the end

Stage 5 sits between pretraining and SFT so you know what the base model could do *before* you changed it. Without that baseline, "did SFT help?" is unanswerable.

### Packaging, conversion, quantization, serving

These four exist because a training artifact and an inference artifact are different things:

| | Training format | Inference format |
|---|---|---|
| File | `model.safetensors` | `model.gguf` |
| Dtype | fp32/fp16 masters + optimizer state | quantized weights, no optimizer |
| Needs | PyTorch, CUDA, Python | a single C++ binary |
| Cares about | gradients, reproducibility | memory bandwidth, latency |

Stage 7 makes the training artifact *portable*. Stage 8 converts it to the inference format. Stage 9 shrinks it. Stage 10 puts it behind an API something else can call.

---

## The two tiers

```
Colab T4  ──[stages 1-8]──▶  🤗 Hub  ──[stages 9-10]──▶  this laptop
   GPU, ephemeral disk        durable        CPU, 2.1 GB free, no torch
```

Two things follow from this split that are worth internalizing:

**1. The Hub is infrastructure, not a portfolio site.** It's how artifacts move between machines that never talk to each other directly. Mid-training checkpoints go there too (stage 4), because free-tier Colab runtimes get reclaimed without warning and a checkpoint on ephemeral disk is not a checkpoint.

**2. The local tier never needs PyTorch.** GGUF conversion happens on Colab specifically so the laptop only ever handles the inference format. This is the same reason production inference servers don't ship a training stack.

---

## What "industry pipeline" means here, and where we simplify

This project is structurally faithful to how models are actually built, at 1/100,000th the scale. Where it diverges, it's worth knowing:

| Real pipeline | Here | Why the difference |
|---|---|---|
| Trillions of tokens, filtered + deduplicated web text | 328M tokens of clean synthetic stories | Data cleaning is its own discipline; TinyStories is pre-clean so we can focus on the pipeline |
| Thousands of GPUs, FSDP/tensor parallelism | One T4 | Distributed training is a large topic that doesn't change the *shape* of the lifecycle |
| SFT → reward model → RLHF/DPO | SFT only | Preference tuning needs preference data and a much stronger base model to be meaningful |
| Benchmark suites (MMLU, HumanEval, …) | Perplexity + qualitative samples | Standard benchmarks are all far above this model's ability; they'd return noise |
| Safety evals, red-teaming, staged rollout | Not covered | Real and important; out of scope for a 15M story model |

The stages we *do* run — tokenizer fitting, architecture, pretraining with checkpointing and LR scheduling, eval, SFT with loss masking, packaging, quantization, serving — are done the way they're really done, not simplified.

---

## The through-line

At the end you should be able to point at a sentence in the VS Code sidebar and trace it all the way back:

```
that sentence
  ← llama-server's sampler
  ← your Q8_0 GGUF
  ← your f16 GGUF
  ← your Hub repo
  ← your SFT run
  ← your pretraining run
  ← 328M tokens through 8 transformer blocks
  ← random initialization
```

Every arrow in that chain is a stage in this repo, and you built all of them.

---

**Next:** [01 — Data](01-data.md)
