"""Stage 7 -- package the trained model into a portable HF repository.

A checkpoint (``.pt``) and a model repo are different artifacts. The checkpoint
exists to resume training: it carries optimizer moments, RNG state, and a step
counter, and it is only meaningful to the code that wrote it. The repo exists to
be *loaded by someone else* -- so it carries weights, architecture, tokenizer,
and documentation, and nothing about how it was produced.

What goes in, and why each file matters:

  model.safetensors        weights. safetensors rather than .bin because
                           loading a pickle executes arbitrary code, and a
                           model repo is by definition untrusted input
  config.json              architecture. This is what tells transformers, and
                           later llama.cpp, that it is looking at a Llama model
  generation_config.json   default sampling parameters, so `generate()` behaves
                           sensibly without the caller knowing the details
  tokenizer.model          the SentencePiece vocabulary. Stage 8 depends on
                           this exact filename existing -- see tokenizer.py
  tokenizer_config.json    special tokens and the chat template
  README.md                the model card
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import torch

from tinyllm import config as cfgmod
from tinyllm.config import PROJECT_NAME, gen_cfg, hub, model_cfg, tok_cfg, train_cfg
from tinyllm.tokenizer import assert_gguf_ready, build_hf_tokenizer


# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------

def export_model(model, sp_model_path: Path, out_dir: Path, chat_model: bool = False,
                 eval_report: dict | None = None) -> Path:
    """Write a complete, loadable HF model repository to ``out_dir``."""
    out_dir = Path(out_dir)
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    model = model.to("cpu").eval()
    model.save_pretrained(str(out_dir), safe_serialization=True)

    build_hf_tokenizer(Path(sp_model_path), out_dir, chat_model=chat_model)

    _write_generation_config(out_dir, chat_model=chat_model)
    _write_training_metadata(out_dir, eval_report)
    write_model_card(out_dir, chat_model=chat_model, eval_report=eval_report)

    # Fail here rather than in stage 8: without tokenizer.model, conversion
    # falls through to the hashed BPE path and dies on an unknown pre-tokenizer.
    assert_gguf_ready(out_dir)

    return out_dir


def _write_generation_config(out_dir: Path, chat_model: bool) -> None:
    """Defaults baked into the repo so callers get sane sampling for free.

    For the chat model, EOS is ``<|im_end|>``: generation must stop at the end of
    the assistant's turn, not at the end of a document.
    """
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(str(out_dir))
    eos_id = tok.convert_tokens_to_ids(tok_cfg.im_end if chat_model else tok_cfg.eos_piece)

    cfg = {
        "bos_token_id": tok_cfg.bos_id,
        "eos_token_id": eos_id,
        "pad_token_id": tok_cfg.pad_id,
        "do_sample": True,
        "temperature": gen_cfg.temperature,
        "top_p": gen_cfg.top_p,
        "top_k": gen_cfg.top_k,
        "repetition_penalty": gen_cfg.repetition_penalty,
        "max_new_tokens": gen_cfg.max_new_tokens,
        "transformers_version": _tf_version(),
    }
    (out_dir / "generation_config.json").write_text(json.dumps(cfg, indent=2), encoding="utf-8")


def _write_training_metadata(out_dir: Path, eval_report: dict | None) -> None:
    """Full config plus eval numbers, so any artifact traces back to its run."""
    meta = {"config": cfgmod.as_dict()}
    if eval_report:
        meta["eval"] = {k: v for k, v in eval_report.items() if k not in ("samples", "temperature_sweep")}
    (out_dir / "training_metadata.json").write_text(json.dumps(meta, indent=2, default=str), encoding="utf-8")


def _tf_version() -> str:
    try:
        import transformers
        return transformers.__version__
    except Exception:
        return "unknown"


# ---------------------------------------------------------------------------
# Model card
# ---------------------------------------------------------------------------

def write_model_card(out_dir: Path, chat_model: bool, eval_report: dict | None = None) -> Path:
    """Write README.md.

    A model card is not decoration. It is the only place a stranger can learn
    what the model was trained on, what it is for, and -- most importantly --
    where it fails. The limitations section here is deliberately blunt.
    """
    pb = model_cfg.param_breakdown()
    kind = "instruction-tuned" if chat_model else "base (text completion)"

    metrics = ""
    if eval_report:
        metrics = (
            "\n## Evaluation\n\n"
            "| metric | value |\n|---|---|\n"
            f"| validation loss | {eval_report.get('val_loss', float('nan')):.4f} |\n"
            f"| validation perplexity | {eval_report.get('val_perplexity', float('nan')):.2f} |\n"
            f"| bits per token | {eval_report.get('bits_per_token', float('nan')):.3f} |\n"
            f"| eval tokens | {eval_report.get('eval_tokens', 0):,} |\n\n"
            "Perplexity is measured on the held-out TinyStories validation split with this\n"
            "model's own tokenizer, so it is comparable across checkpoints of this model and\n"
            "**not** comparable to any model with a different vocabulary.\n"
        )

    usage = _CHAT_USAGE if chat_model else _BASE_USAGE

    card = f"""---
license: cdla-sharing-1.0
datasets:
  - roneneldan/TinyStories{"" if not chat_model else chr(10) + "  - roneneldan/TinyStoriesInstruct"}
language:
  - en
pipeline_tag: text-generation
tags:
  - llama
  - tiny
  - educational
  - gguf
---

# {PROJECT_NAME} — {kind}

A {pb['total'] / 1e6:.1f}M-parameter Llama-architecture language model trained from
random initialization on [TinyStories](https://huggingface.co/datasets/roneneldan/TinyStories).

Built as a complete walk through the model lifecycle — tokenizer, architecture,
pretraining, evaluation, instruction tuning, packaging, quantization, and local
serving. It is small enough to train in about 45 minutes on a free Colab T4 and
to run on a 2-core laptop CPU with no GPU.

## Architecture

| | |
|---|---|
| Parameters | {pb['total']:,} ({pb['non_embedding']:,} non-embedding) |
| Layers | {model_cfg.num_hidden_layers} |
| Hidden size | {model_cfg.hidden_size} |
| Attention heads | {model_cfg.num_attention_heads} query / {model_cfg.num_key_value_heads} key-value (GQA) |
| Head dim | {model_cfg.head_dim} |
| MLP | SwiGLU, intermediate {model_cfg.intermediate_size} |
| Normalization | RMSNorm (eps {model_cfg.rms_norm_eps}) |
| Position encoding | RoPE (theta {model_cfg.rope_theta:.0f}) |
| Context length | {model_cfg.max_position_embeddings} |
| Vocabulary | {model_cfg.vocab_size} (SentencePiece BPE, byte fallback) |
| Embeddings | tied input/output |

## Training

| | |
|---|---|
| Tokens | {train_cfg.total_tokens / 1e6:.0f}M (~{train_cfg.chinchilla_ratio(model_cfg):.0f} per parameter) |
| Steps | {train_cfg.max_steps:,} at {train_cfg.tokens_per_step:,} tokens/step |
| Optimizer | AdamW (betas {train_cfg.beta1}/{train_cfg.beta2}, wd {train_cfg.weight_decay} on matrices only) |
| Schedule | cosine, {train_cfg.warmup_steps} warmup steps, peak LR {train_cfg.learning_rate} |
| Precision | fp16 AMP with loss scaling |
| Hardware | 1x NVIDIA T4 (Colab free tier) |
{metrics}
## Usage

{usage}

### With llama.cpp

GGUF conversions are included in this repo.

```bash
llama-server -m {PROJECT_NAME}-Q8_0.gguf -c {model_cfg.max_position_embeddings} --host 127.0.0.1 --port 8080
```

## Limitations

This model has {pb['total'] / 1e6:.1f}M parameters and a {model_cfg.vocab_size}-token vocabulary,
trained exclusively on synthetic children's stories. Be concrete about what that means:

- **It only does one thing.** It writes simple short stories in the TinyStories
  style. Anything else — code, arithmetic, factual questions, translation,
  summarization of arbitrary text — produces confident nonsense.
- **Its vocabulary is small.** Words outside a children's-story vocabulary fall
  back to individual bytes, which it handles poorly.
- **Context is {model_cfg.max_position_embeddings} tokens.** There is no long-range coherence to be had.
- **No safety tuning of any kind.** It has had no alignment work beyond
  instruction tuning on story prompts.
- **Quantization hurts more than usual.** Small models have less parameter
  redundancy to absorb rounding error; Q4_K_M is measurably worse here than the
  usual "negligible loss" guidance for 7B+ models would suggest.

Not suitable for any production use. It is a teaching artifact.

## Training data

[TinyStories](https://huggingface.co/datasets/roneneldan/TinyStories) — synthetic
short stories generated by GPT-3.5/GPT-4, constrained to the vocabulary of a
3-4 year old. Licensed CDLA-Sharing-1.0.
{"" if not chat_model else chr(10) + "Instruction tuning used [TinyStoriesInstruct](https://huggingface.co/datasets/roneneldan/TinyStoriesInstruct)." + chr(10)}
"""
    path = Path(out_dir) / "README.md"
    path.write_text(card, encoding="utf-8")
    return path


_BASE_USAGE = """```python
from transformers import AutoModelForCausalLM, AutoTokenizer

tok = AutoTokenizer.from_pretrained("REPO_ID")
model = AutoModelForCausalLM.from_pretrained("REPO_ID")

ids = tok("Once upon a time, there was a little girl named Lily.", return_tensors="pt")
out = model.generate(**ids, max_new_tokens=200, do_sample=True, temperature=0.8)
print(tok.decode(out[0], skip_special_tokens=True))
```"""

_CHAT_USAGE = """```python
from transformers import AutoModelForCausalLM, AutoTokenizer

tok = AutoTokenizer.from_pretrained("REPO_ID")
model = AutoModelForCausalLM.from_pretrained("REPO_ID")

messages = [{"role": "user", "content": "Write a story about a lost puppy."}]
prompt = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
ids = tok(prompt, return_tensors="pt")
out = model.generate(**ids, max_new_tokens=250, do_sample=True, temperature=0.8)
print(tok.decode(out[0], skip_special_tokens=True))
```"""


# ---------------------------------------------------------------------------
# The stage 7 gate
# ---------------------------------------------------------------------------

def verify_export(out_dir: Path, reference_model=None, device: str = "cpu") -> dict:
    """Reload the exported repo from disk and prove it is the same model.

    Round-tripping through ``from_pretrained`` is what catches a dtype change, a
    dropped tied weight, or a config field that did not serialize. If this passes,
    the repo works for anyone who downloads it.
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer

    out_dir = Path(out_dir)
    tok = AutoTokenizer.from_pretrained(str(out_dir))
    reloaded = AutoModelForCausalLM.from_pretrained(str(out_dir)).to(device).eval()

    result = {
        "n_params": sum(p.numel() for p in reloaded.parameters()),
        "vocab_size": len(tok),
        "has_chat_template": tok.chat_template is not None,
        "files": sorted(p.name for p in out_dir.iterdir()),
    }

    for required in ("tokenizer.model", "config.json", "generation_config.json", "README.md"):
        if required not in result["files"]:
            raise AssertionError(f"{required} missing from the exported repo")

    if reference_model is not None:
        ref = reference_model.to(device).eval()
        ids = torch.randint(0, model_cfg.vocab_size, (1, 32))
        with torch.no_grad():
            a = ref(input_ids=ids).logits
            b = reloaded(input_ids=ids).logits
        diff = (a - b).abs().max().item()
        if diff > 1e-4:
            raise AssertionError(f"reloaded model differs from the original: max|delta|={diff:.3e}")
        result["max_logit_diff"] = diff

    return result


# ---------------------------------------------------------------------------
# Hub
# ---------------------------------------------------------------------------

def push_to_hub(out_dir: Path, repo_id: str | None = None, private: bool = False,
                commit_message: str = "Upload model") -> str:
    """Upload the repo folder. Returns the model page URL."""
    from huggingface_hub import HfApi

    repo_id = repo_id or hub.model_repo
    if repo_id.startswith("CHANGEME/"):
        raise ValueError(
            "HubConfig.user is still 'CHANGEME'. Set it in src/tinyllm/config.py "
            "to your Hugging Face username before pushing."
        )

    api = HfApi()
    api.create_repo(repo_id, repo_type="model", exist_ok=True, private=private)

    # README.md ships with REPO_ID as a placeholder; substitute it now that the
    # destination is known, so the usage snippets are copy-pasteable.
    readme = Path(out_dir) / "README.md"
    if readme.exists():
        readme.write_text(readme.read_text(encoding="utf-8").replace("REPO_ID", repo_id), encoding="utf-8")

    api.upload_folder(folder_path=str(out_dir), repo_id=repo_id, repo_type="model",
                      commit_message=commit_message)
    return f"https://huggingface.co/{repo_id}"


def upload_gguf(gguf_path: Path, repo_id: str | None = None) -> str:
    """Add a GGUF file to the model repo alongside the safetensors.

    Both formats in one repo is the convention: training-format weights for
    anyone who wants to fine-tune, GGUF for anyone who wants to run it.
    """
    from huggingface_hub import HfApi

    repo_id = repo_id or hub.model_repo
    api = HfApi()
    api.create_repo(repo_id, repo_type="model", exist_ok=True)
    api.upload_file(
        path_or_fileobj=str(gguf_path),
        path_in_repo=Path(gguf_path).name,
        repo_id=repo_id,
        repo_type="model",
        commit_message=f"Add {Path(gguf_path).name}",
    )
    return f"https://huggingface.co/{repo_id}/blob/main/{Path(gguf_path).name}"
