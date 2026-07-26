# 09 — Quantization and serving

**Notebooks:** `notebooks/local/09_inspect_gguf.ipynb`, `10_quant_tradeoff.ipynb`
**Scripts:** `scripts/quantize.ps1`, `scripts/serve.ps1`

Everything from here runs on your laptop, with **no PyTorch**.

---

## What quantization is

Storing weights in fewer bits. A weight matrix is split into blocks; each block
gets a scale factor, and individual weights are stored as small integers to be
multiplied by that scale.

```
f16:    [0.0234, -0.0891, 0.0445, ...]     16 bits each
Q8_0:   scale=0.0007, [33, -127, 63, ...]   8 bits each + one scale per 32
```

The saving is real and the cost is rounding error. The question is always how much
error, and this stage **measures it rather than assuming**.

## The K-quants

| level | bits/weight | scheme |
|---|---|---|
| f16 | 16 | reference |
| Q8_0 | 8 | one scale per block of 32, uniform |
| Q5_K_M | ~5.5 | K-quant: sub-block scales, **mixed** precision |
| Q4_K_M | ~4.5 | K-quant, more aggressive |

"K-quants" beat flat schemes in two ways:

1. **Hierarchical scales** — a super-block scale plus per-sub-block scales, so a
   few outlier weights don't wreck the resolution for their neighbours.
2. **Mixed precision** — the `_M` variants give higher precision to the tensors
   where error propagates furthest (attention output, and the feed-forward
   down-projection), and lower precision elsewhere.

That second point is why `Q4_K_M` is meaningfully better than a flat 4-bit scheme
at essentially the same file size. Not all weights matter equally.

---

## The claim you'll read everywhere, and why we test it

> "Q4_K_M is the sweet spot — near-lossless at a quarter the size."

That advice is written for **7B+ models**, and for those it's roughly true. It is
not a law of nature.

A larger model has enormous **parameter redundancy**: many weights encode
overlapping information, so rounding each to 4 bits loses little the others can't
cover. A **15.7M-parameter** model has far less slack — every weight is doing more
work, so the same rounding error costs more.

**Expect Q4_K_M to hurt noticeably here.** Notebook 10 measures it on your model
with `llama-perplexity` over held-out TinyStories, and plots quality against size.

The general lesson is about method, not this model: **quantization level is a
decision to make from a measurement on the model you actually have.** The specific
numbers you get are less important than the habit of getting them.

### Why held-out text matters here too

Measuring perplexity on training data would tell you how well the model memorized,
and quantization damage would be partly masked by that memorization. The notebook
pulls 100 rows from the *validation* split via the Hub's datasets-server API — a
few hundred KB, no 7.6 GB download.

## Practical guidance

- **Q8_0** is the safe default and what `serve.ps1` uses. Near-indistinguishable
  from f16 at ~53% the size.
- **Q4_K_M** is where to look hardest at small scale.
- On a CPU, smaller weights also mean **fewer bytes to move**. Inference at this
  size is memory-bandwidth bound rather than compute bound, so quantization
  usually buys latency as well as disk.

---

## Serving

```powershell
.\scripts\serve.ps1
```

`llama-server` implements the OpenAI API — `/v1/chat/completions`,
`/v1/completions`, plus a browser UI at the root URL. That single fact is what
makes stage 10 short: the `openai` SDK, Continue.dev and `curl` all work against
it with nothing but a base-URL change.

### Flags that matter

| flag | why |
|---|---|
| `--jinja` | use the chat template **embedded in the GGUF** rather than a built-in guess. Without it a ChatML model can be served with the wrong turn format and quietly produce worse output |
| `--ctx-size 512` | must not exceed `max_position_embeddings`. Asking for more than the model was trained for produces garbage past the training length |
| `--threads 4` | this laptop has 2 physical cores / 4 logical |
| `--alias tinyllm` | the model id clients request |

`--jinja` is the one people miss. It's the fourth and final hop of the chat
template journey that started in `config.py`.

---

## The local disk budget

This machine had ~2.1 GB free at the start. The whole point of the two-tier design
was fitting inside it:

| | size |
|---|---|
| Python venv (no torch) | ~350 MB |
| llama.cpp binaries | ~60 MB extracted (17 MB zipped) |
| GGUF set (f16 + 3 quants) | ~120 MB |
| Continue extension | ~50 MB |
| **total** | **~580 MB** |

For comparison, a PyTorch CPU install alone is ~1.2 GB unpacked — it would have
consumed most of the free space before a single model file existed. Building
llama.cpp from source instead of using the prebuilt release would have added
several GB of MSVC Build Tools on top.

Neither constraint changed *what* the project does; both changed *where* each
stage runs. That's the tradeoff production systems make constantly.

## Gate

- [x] GGUF metadata matches `config.py`
- [x] Tokenizer and chat template confirmed inside the file
- [x] Perplexity measured across all quantization levels
- [x] Quality/size curve plotted and checked against readable output
- [x] Server responding on `/v1/chat/completions`

**Next:** [10 — VS Code](10-vscode.md)
