# 06 — Instruction tuning

**Notebook:** `notebooks/colab/06_sft.ipynb`
**Code:** `src/tinyllm/sft.py`

## What this stage does

Turns a text completer into something that responds to a request. ~1,500 steps,
~10 minutes.

## The problem it solves

After pretraining, the model continues text. Given "Once upon a time" it writes a
story. Given "Write a story about a puppy" it will happily continue *that
sentence* — "…and a kitten who lived in a barn" — rather than obey it.

Nothing has ever taught it that some text is an instruction to be carried out
rather than prose to be extended. That's what SFT does, and it's cheap: ~1,500
steps at a quarter of the pretraining batch size. It only works because
pretraining was expensive.

---

## Chat templates

Turns get wrapped in ChatML so the model can distinguish "what I was asked" from
"what I should say":

```
<|im_start|>user
Write a story about a lost puppy.<|im_end|>
<|im_start|>assistant
Once upon a time...<|im_end|>
```

**The same template string takes four hops:**

```
config.py :: TokenizerConfig.chat_template
  → tokenizer_config.json          (stage 2 / 7)
  → GGUF metadata                  (stage 8)
  → llama-server request formatting (stage 9)
```

If it disagrees anywhere along that chain, the served model receives prompts in a
format it was never trained on, and gets quietly worse for no visible reason.
There's no error — just degraded output that's very hard to attribute.

Each stage in this project verifies its own hop:
notebook 02 asserts the control tokens are atomic, notebook 07 renders the
template and checks for a generation prompt, notebook 09 reads it back out of the
GGUF, and `serve.ps1` passes `--jinja` so the server uses the embedded template
rather than guessing.

**Why ChatML specifically:** it's widely recognised, and llama.cpp detects it
automatically. A custom format would work too, but you'd be responsible for
teaching every downstream tool about it.

---

## Completion-only loss masking

Prompt tokens get label `-100` (PyTorch's `ignore_index`) so they contribute no
gradient:

```
<|im_start|>user\nWrite a story...<|im_end|>\n<|im_start|>assistant\n
└────────────────── masked, label = -100 ──────────────────────────┘
Once upon a time, there was a puppy...<|im_end|>
└──────────── supervised, real labels ────────────┘
```

Without this, the model spends capacity learning to *generate instructions*, which
isn't the job.

**This fails silently.** Training still converges, the loss still falls — to a
worse model. There's no error and no obvious symptom. So `describe_masking()`
decodes a real example and asserts:

- the number of masked tokens equals the prompt length
- no prompt token contributes to the loss
- not everything is masked (which would produce zero gradient)

The notebook prints the token-by-token boundary. It's a thirty-second check
against a bug that is otherwise invisible.

### Why prompt and completion are encoded separately

```python
prompt_ids = [bos] + sp.EncodeAsIds(prompt_text)
completion_ids = sp.EncodeAsIds(completion_text)
input_ids = prompt_ids + completion_ids
```

Not: encode the joined string and slice at `len(prompt_ids)`.

BPE can merge across the boundary, so the prompt's token count *within the joined
string* need not equal its count alone — and the mask would land in the wrong
place by a token or two. Encoding separately guarantees the model sees exactly
what we masked.

---

## Instruction synthesis

`TinyStoriesInstruct` gives metadata (`Features`, `Words`, `Summary`) and a story.
Instructions are built from that metadata using **sampled templates**:

```python
"Write a story using the words: {words}."
"Write a short story that includes these words: {words}."
"Tell me a story with the words {words} in it."
```

A single fixed template produces a model that breaks the moment a user phrases the
request differently — it learns the sentence, not the task. Varying the surface
form is what makes the behaviour generalize.

### The dataset format

`TinyStoriesInstruct` is stored **one line per row**, not one example per row.
Records are separated by `<|endoftext|>`, and header fields appear in **arbitrary
order**:

```
Features: Dialogue                 Summary: Lily steals a bike...
Words: quit, oak, gloomy           Words: ride, work, upset
Summary: Sara and Ben...           Features: Dialogue, BadEnding
Story:                             Story:
<blank>                            <blank>
Sara and Ben were playing...       Lily liked to ride her bike...
<|endoftext|>                      <|endoftext|>
```

So `iter_instruct_records()` accumulates lines into records and matches headers by
prefix, not position. Treating rows as examples — the obvious first guess — yields
garbage.

## Over-long examples are dropped, not truncated

Truncating would cut stories off mid-sentence and teach the model to stop
arbitrarily. Dropping a few percent of examples is cheaper than that damage.

## Learning rate: 6× below pretraining

`1e-4` vs `6e-4`. This stage reshapes output format; it isn't meant to relearn
language. Too high an LR here causes **catastrophic forgetting** — the model gets
worse at the one thing it was good at.

---

## The metric that gets worse, and why that's correct

After SFT, **perplexity on raw TinyStories almost certainly goes up** — and the
model is far more useful.

That isn't a contradiction. The model was moved off the distribution perplexity
measures (raw story text) and onto one it doesn't (instruction-formatted
dialogue). The metric is now measuring the wrong thing.

This is the sharpest illustration in the project of why you don't optimize a
number without asking what it tracks. **Had you been tuning against validation
perplexity, you would have concluded SFT was harmful and discarded the step that
made the model usable.**

## What a full alignment pipeline adds

SFT is the first of three stages in a typical post-training pipeline:

1. **SFT** — teach the response format. (What we do.)
2. **Reward modelling** — train a model to score responses from human preference
   pairs.
3. **RLHF / DPO** — optimize the policy against those preferences.

We stop at 1. Preference tuning needs preference data and a base model strong
enough for the differences between two responses to be meaningful. At 15.7M
parameters, both are out of reach — [DPO](https://arxiv.org/abs/2305.18290) on
this model would be fitting noise.

## Gate

- [x] Loss mask verified by decoding a real batch
- [x] Instructions synthesised from varied templates
- [x] Over-long examples dropped, not truncated
- [x] Model responds to instructions instead of continuing them
- [x] Understood why validation perplexity got worse

**Next:** [07 — Packaging and the Hub](07-hub.md)
