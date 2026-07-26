# 02 — Tokenizer

**Notebook:** `notebooks/colab/02_tokenizer.ipynb`
**Code:** `src/tinyllm/tokenizer.py`

## What this stage does

Fits an 8192-piece SentencePiece BPE vocabulary to the corpus, verifies it
round-trips exactly, and packs the whole dataset into a flat `uint16` token array.

---

## Vocabulary size is a model dimension

This is the part people underestimate. `vocab_size` doesn't just affect the
tokenizer — it sets the size of the embedding table, which at small scale
dominates everything:

| vocab | embedding params | share of a ~15.7M model |
|---|---|---|
| 8,192 | 3.1M | 20% |
| 32,000 (Llama 2) | 12.3M | 49% |
| 128,256 (Llama 3) | 49.2M | 76% — larger than the rest combined |

Reusing an off-the-shelf tokenizer would have spent most of the parameter budget
on embeddings for tokens this corpus never contains. Those rows would receive
almost no gradient and stay near their initialization forever.

There's a real tension here, and it flips with scale:

- **Larger vocabulary** → fewer tokens per document → more text fits in the
  context, and each forward pass covers more material.
- **Smaller vocabulary** → smaller embedding table → more parameters left for the
  layers that actually compute.

At 15.7M parameters the second consideration wins decisively. At 7B the first
does, which is why large models use 32k–256k vocabularies. Neither choice is
universally right, and this is the clearest example in the project of a
hyperparameter whose correct value depends on the rest of the design.

---

## The GGUF landmine

**This is the most important paragraph in the docs.**

`convert_hf_to_gguf.py` resolves a Llama model's vocabulary like this:

```python
try:
    self._set_vocab_sentencepiece()      # looks for a file named tokenizer.model
except FileNotFoundError:
    try:
        self._set_vocab_llama_hf()
    except (FileNotFoundError, TypeError):
        self._set_vocab_gpt2()           # hashes the pre-tokenizer
```

That last path computes a **hash of the pre-tokenizer configuration** and compares
it against a hardcoded registry in `get_vocab_base_pre()`. If your tokenizer
isn't in that registry, conversion dies:

```
NotImplementedError: BPE pre-tokenizer was not recognized -
please update get_vocab_base_pre()
```

Every custom `tokenizers`-library BPE ever trained hits this
([#8649](https://github.com/ggml-org/llama.cpp/issues/8649),
[#9927](https://github.com/ggml-org/llama.cpp/issues/9927)). The registry only
contains hashes for *published* models, because someone had to run
`convert_hf_to_gguf_update.py` and commit the result.

**We avoid it entirely by training SentencePiece instead.** Because
`_set_vocab_sentencepiece()` is tried *first* and does no hash lookup, shipping a
file literally named `tokenizer.model` means the failing path is never reached.
This is the route TinyLlama and most Llama-derivative models take.

`tokenizer.assert_gguf_ready()` checks for that file at the end of stage 2 and
again before conversion in stage 7 — failing in stage 2 costs seconds, failing in
stage 8 costs a re-export.

### Side quest: trigger it deliberately

You own this tokenizer, so this is the cheapest possible way to see the machinery:

1. Train a `tokenizers`-library BPE on the same corpus, save as `tokenizer.json`.
2. Export a model repo **without** `tokenizer.model`.
3. Run `convert_hf_to_gguf.py` → `NotImplementedError`.
4. Fix it: run `convert_hf_to_gguf_update.py` with your model added to its list,
   or add your hash to `get_vocab_base_pre()` by hand.

Doing this once makes the failure mode legible forever, and it's the kind of
problem that is genuinely hard to debug the first time you hit it in anger.

---

## The training options, and why each is non-default

```python
byte_fallback=True              # any byte is representable -- never lossy
normalization_rule_name="identity"   # no NFKC, so decode(encode(s)) == s
remove_extra_whitespaces=False  # whitespace survives round-tripping
split_digits=True               # "1997" doesn't become an atom unrelated to "1998"
user_defined_symbols=[...]      # ChatML tokens stay atomic
```

**`byte_fallback`** costs 256 vocab slots and buys the guarantee that no input is
ever unrepresentable. Without it, an unseen character becomes `<unk>` and is
permanently lost.

**`identity` normalization + no whitespace stripping** are what make the
round-trip test in the notebook meaningful. With SentencePiece defaults, NFKC
normalization silently rewrites the text and `decode(encode(s)) != s`. That
difference is invisible during training and shows up as strange behaviour on
punctuation and accented characters at inference time.

**`user_defined_symbols`** keeps `<|im_start|>` and `<|im_end|>` as single ids. If
they split into subwords, the chat template stops delimiting turns: the model sees
fragments rather than a boundary, and `llama-server` never finds an EOS. Stage 6
depends on this and the notebook asserts it.

---

## Why the packed format is `uint16` and flat

- **`uint16`** — 8192 fits in 16 bits. As `int64` the corpus would be 2.6 GB; as
  `uint16` it's 656 MB, and it memory-maps cleanly.
- **Flat, not per-document** — training samples random windows of `seq_len + 1`
  from one contiguous array. No padding, no ragged batches, no wasted compute.
- **EOS between documents** — windows that straddle a boundary teach the model
  where stories end. Omit this and it never learns to stop, which appears at
  inference as generation running until the token limit.

## Gate

- [x] Round-trip exact on 1,000 held-out documents
- [x] ChatML control tokens are single atomic ids
- [x] `tokenizer.model` on disk — stage 8 takes the SentencePiece path
- [x] `train.bin` / `val.bin` packed, validation uncontaminated
- [x] Tokenizer pushed to the Hub

**Next:** [03 — Architecture](03-architecture.md)
