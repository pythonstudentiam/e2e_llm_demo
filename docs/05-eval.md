# 05 — Evaluation

**Notebook:** `notebooks/colab/05_eval.ipynb`
**Code:** `src/tinyllm/evaluate.py`

## What this stage does

Measures the base model, so that after instruction tuning you can answer "did that
help?" with evidence rather than impressions.

---

## Perplexity: what it is and what it isn't

```
perplexity = exp(mean cross-entropy)
```

Loosely: how many tokens the model is effectively choosing between at each step. A
perplexity of 20 means it's about as uncertain as if it were picking uniformly
from 20 options.

**A random model scores `vocab_size`.** Ours starts at 8192 and should end
somewhere in the low tens.

### What perplexity is good for

Comparing checkpoints **of the same model with the same tokenizer**. That's the
use here: tracking progress across training, and measuring what quantization costs
in stage 9.

### What it is not good for

**Comparing different models.** Perplexity is per-token, and tokens differ between
tokenizers. A model with a smaller vocabulary gets a lower perplexity essentially
for free, because it's choosing from fewer options at each step — and because its
tokens are shorter, it gets more predictions per unit of text, each individually
easier.

This is why leaderboards stopped reporting perplexity. If you ever see two models
with different tokenizers compared on it, the comparison is meaningless.

**Bits per token** (`loss / ln 2`) is the same quantity in compression units, and
it's a useful reframing: a language model *is* a compressor. A model at 4
bits/token could encode this corpus at 4 bits per token where a naive scheme needs
`log₂(8192) = 13`. Learning and compression are the same thing viewed from two
directions — the connection [Shannon](https://en.wikipedia.org/wiki/Entropy_(information_theory))
made and that modern LLM training still rests on.

---

## Repetition metrics

The failure mode small models actually have is **looping** — the model finds a
phrase and repeats it forever.

**Perplexity barely moves when this happens.** Each repeated token is individually
very predictable, so the model is confident and the loss is low. This is the
cleanest example in the project of a metric looking healthy while the output is
unusable.

`distinct-n` = unique n-grams / total n-grams:

- **1.0** — nothing ever repeats
- **~0.7–0.9** — normal for natural text
- **below ~0.6 for distinct-3** — usually reads as broken

Always measure this alongside perplexity. They fail in different directions.

---

## Sampling parameters

| parameter | effect |
|---|---|
| `temperature` | divides logits before softmax. <1 sharpens (coherent, repetitive), >1 flattens (varied, incoherent) |
| `top_k` | keep only the k most likely tokens |
| `top_p` (nucleus) | keep the smallest set whose cumulative probability ≥ p |
| `repetition_penalty` | down-weight tokens already generated |

The temperature sweep in the notebook shows the tradeoff directly. **Where the
sweet spot sits is a property of this model, not a universal constant** — a
stronger model tolerates higher temperature before losing coherence, because its
distribution is sharper to begin with. Measure it; don't inherit someone else's
default.

`top_p` and `top_k` both exist because pure temperature sampling occasionally
draws from the far tail and derails a generation. Truncating the tail is cheap
insurance.

---

## What we deliberately don't measure

MMLU, HumanEval, GSM8K, HellaSwag — all of them would return **noise** on a 15.7M
model trained on children's stories. Most are multiple-choice or code tasks far
outside what this model can represent at all.

Running them to produce a number would be worse than not running them, because the
number would look like information. A benchmark below the noise floor tells you
nothing except that you ran it.

This generalizes: **a metric is only meaningful when the model is in a regime
where it discriminates.** Applying a frontier-model benchmark to a tiny model is
the same error as applying perplexity across tokenizers.

## What a real pipeline adds here

- **Held-out benchmark suites** appropriate to the model's scale and domain
- **Human evaluation** — still the gold standard for open-ended generation
- **LLM-as-judge** — a stronger model scoring outputs against a rubric; cheap and
  correlates reasonably with human judgment
- **Red-teaming and safety evals** before any release
- **Regression tracking** across training runs, so a change that helps one metric
  and quietly breaks another gets caught

## Gate

- [x] Perplexity measured on uncontaminated held-out data
- [x] Repetition characterised across temperatures
- [x] Baseline saved for the stage 6 comparison

**Next:** [06 — Instruction tuning](06-sft.md)
