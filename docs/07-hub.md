# 07 — Packaging and the Hub

**Notebook:** `notebooks/colab/07_package_push.ipynb`
**Code:** `src/tinyllm/export.py`

## What this stage does

Turns a training checkpoint into an artifact a stranger can load.

## A checkpoint and a model repo are different things

| | checkpoint (`.pt`) | model repo |
|---|---|---|
| Purpose | resume training | be loaded by someone else |
| Contains | weights, optimizer moments, RNG state, step | weights, architecture, tokenizer, docs |
| Meaningful to | the code that wrote it | anyone |
| Size here | ~190 MB | ~63 MB |

The checkpoint is 3× larger because Adam stores two moment tensors per parameter —
state that is worthless the moment training ends.

## What goes in, and why each file matters

| file | why |
|---|---|
| `model.safetensors` | weights |
| `config.json` | architecture — what tells transformers, and later llama.cpp, this is a Llama model |
| `generation_config.json` | default sampling params, so `generate()` behaves sensibly without the caller knowing details |
| `tokenizer.model` | the SentencePiece vocabulary. **Stage 8 depends on this exact filename** |
| `tokenizer_config.json` | special tokens and the chat template |
| `README.md` | the model card |
| `training_metadata.json` | full config + eval numbers, so the artifact traces back to its run |

### safetensors, not pickle

`.bin` checkpoints are Python pickles, and **unpickling executes arbitrary code**.
A model repo is by definition untrusted input — you're downloading a file a
stranger uploaded — so this is a genuine remote-code-execution vector, not a
theoretical one.

[safetensors](https://github.com/huggingface/safetensors) is a flat format: a JSON
header of names, dtypes and offsets, then raw tensor bytes. Nothing to execute. It
also memory-maps, so loading is faster and doesn't spike RAM.

This is the reason the format switched ecosystem-wide, and it's worth
understanding rather than treating as a default.

### The EOS decision

`chat_model=True` sets the tokenizer's EOS to `<|im_end|>` rather than `</s>`.

This matters at serving time: `llama-server` stops generating at the GGUF's EOS
id, and for a ChatML model that has to be the **turn terminator**, not the
end-of-document token. Get it wrong and the server runs past the end of the reply
until it hits the token cap — visible as `finish_reason: length` on every request.

Notebook 11 checks for `finish_reason: stop`, which is this decision working.

---

## Model cards

A model card is not decoration. It's the only place someone can learn what the
model was trained on, what it's for, and where it fails.

Ours states bluntly that the model does exactly one thing and produces confident
nonsense outside it. That's true, and **a card that oversells is worse than no
card** — it transfers the cost of discovering the limitations onto whoever uses
the model, usually after they've built something on it.

The YAML frontmatter is machine-readable and drives Hub features:

```yaml
license: cdla-sharing-1.0
datasets: [roneneldan/TinyStories]
pipeline_tag: text-generation
tags: [llama, tiny, educational, gguf]
```

Getting `license` right matters — TinyStories is CDLA-Sharing-1.0, and a model
trained on it inherits obligations. Licensing of model weights derived from
licensed data is genuinely unsettled law; stating the provenance accurately is the
minimum.

---

## The Hub as infrastructure

In this project the Hub isn't a portfolio site — it's the **transport between
tiers**. Colab has a GPU and no persistence; the laptop has persistence and no
GPU. They never talk to each other directly.

| repo | contents | why |
|---|---|---|
| `<user>/tinyllm-checkpoints` | rolling `latest.pt`, tokenizer | survives runtime reclamation; private |
| `<user>/tinyllm` | safetensors + tokenizer + GGUF + card | the published artifact |

Mid-training checkpoints go to a **separate private repo** so the public model
repo stays clean — nobody downloading the model wants a 190 MB optimizer state.

Repos are git (with LFS for large files), so revisions are commits and you can pin
`revision="abc123"` in `from_pretrained` for reproducibility. Branches work as
model versions. This is more useful than it first appears: "which exact weights
produced this result" is a question that comes up constantly and is otherwise
painful to answer.

## The gate: reload from disk

Round-tripping through `from_pretrained` catches a dtype change, a dropped tied
weight, or a config field that didn't serialize. The notebook then goes further
and loads the model **by repo id from the Hub** — which is what a stranger's code
does, and the only way to know the published artifact actually works.

## Gate

- [x] Exported as safetensors with config, tokenizer, generation config
- [x] Reload from disk produces identical logits
- [x] Chat template renders a generation prompt
- [x] Model card written with honest limitations
- [x] Loads from the Hub by repo id

**Next:** [08 — GGUF conversion](08-gguf.md)
