# 08 — GGUF conversion

**Notebook:** `notebooks/colab/08_gguf.ipynb`

## What this stage does

Converts the training-format model into an inference-format model, and **proves
the conversion was lossless**.

## Why a second format exists

| | safetensors (training) | GGUF (inference) |
|---|---|---|
| Needs | PyTorch, Python, CUDA | one C++ binary |
| Layout | tensor blobs + separate JSON config | tensors *and* all metadata in one file |
| Weights | fp32 / fp16 | fp16 or quantized |
| Optimised for | gradients, flexibility | memory bandwidth, mmap, fast startup |
| Tokenizer | separate files | embedded |

GGUF is **self-contained**: architecture, hyperparameters, the entire tokenizer
vocabulary, and the chat template all live inside one file. That's what lets
`llama-server -m model.gguf` work with no config files, no Python, and no
tokenizer sitting beside it.

Contrast with stage 7, where weights, config and tokenizer were three separate
files that had to agree with each other. Every "it works locally but not in
production" story about mismatched tokenizer versions is a consequence of that
split, and GGUF's answer is to make the mismatch structurally impossible.

The format is defined
[here](https://github.com/ggml-org/llama.cpp/blob/master/docs/development/gguf.md).
It replaced the older GGML format in 2023, primarily to add extensible metadata —
which is exactly the property the chat template depends on.

## Why conversion runs on Colab

`convert_hf_to_gguf.py` needs PyTorch to read the safetensors. The local Windows
tier has **no PyTorch by design** — that's what keeps its footprint at ~700 MB on
a machine with ~2 GB free.

So conversion happens on the GPU tier, and the laptop only ever sees the inference
format. This mirrors production: inference servers don't ship a training stack.

## Where stage 2's decision pays off

`convert_hf_to_gguf.py` tries `_set_vocab_sentencepiece()` **first**, which looks
for a file named exactly `tokenizer.model`. We have one, so conversion takes that
path and never reaches `_set_vocab_gpt2()` — the one that hashes the pre-tokenizer
against a hardcoded registry and raises `NotImplementedError` for custom
vocabularies.

See [02-tokenizer.md](02-tokenizer.md) for the full mechanism. The notebook
asserts the file is present before converting, so if the export regressed you find
out immediately rather than reading a confusing stack trace.

---

## The gate: greedy output must match

Run the *same* prompt through `transformers` and through `llama.cpp` with greedy
decoding (temperature 0, so there's no sampling randomness to hide behind). The
token sequences must match.

This is a genuine end-to-end check across **two entirely independent
implementations** — one Python/PyTorch, one C++. If a tensor were transposed, a
rope parameter misread, or the vocabulary misordered, the outputs would diverge.
Very little else in the pipeline gives you a cross-implementation check this
strong.

### Reading a mismatch

- **Diverges from the very first token** — a real conversion bug. Check tensor
  names, the rope parameters, and the vocabulary order in the metadata.
- **Agrees for a while, then drifts** — usually benign fp16 rounding. The two
  implementations accumulate floating-point error differently, and once two logits
  are within rounding distance, greedy decoding picks differently and the
  sequences separate permanently. A few dozen matching tokens is strong evidence
  the conversion is correct.

## What to check in the metadata

```
general.architecture              llama
llama.block_count                 8
llama.embedding_length            384
llama.attention.head_count        6
llama.attention.head_count_kv     2      <- GQA survived
llama.rope.freq_base              10000
tokenizer.ggml.model              llama  <- SentencePiece path taken
tokenizer.ggml.eos_token_id       <im_end>
tokenizer.chat_template           {% for message in messages %}...
```

Notebook 09 asserts these against `config.py` — a mismatch means something drifted
between training and conversion.

**The chat template is hop 3 of 4.** If it's missing, `llama-server` falls back to
a generic format and the model sees prompts unlike anything it was trained on.
Notebook 09 warns loudly if it's absent.

## The tensor count is lower than expected — correctly

`token_embd.weight` appears once but is used both to embed input tokens and to
produce output logits. The tied-embedding decision from stage 3 is visible right
there in the file layout, and the 3.1M-parameter saving is real, not an accounting
trick.

## Gate

- [x] Converted via the SentencePiece path — no pre-tokenizer hash error
- [x] Metadata matches `config.py`
- [x] Chat template present inside the GGUF
- [x] Greedy output matches `transformers`
- [x] f16 GGUF published to the Hub

**Next:** [09 — Quantization and serving](09-quantization.md)
