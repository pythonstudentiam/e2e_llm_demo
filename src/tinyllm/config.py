"""Single source of truth for every hyperparameter in the pipeline.

This module is deliberately dependency-free (stdlib only). It is imported both
on the Colab GPU tier -- where torch and transformers exist -- and on the local
Windows tier, where they deliberately do not. Keep it that way: any import of
torch/transformers here must be lazy, inside a function body.

Change the numbers here and the entire pipeline follows: tokenizer vocab,
model shape, token budget, and step count are all derived from this file.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, asdict
from typing import Any

# ---------------------------------------------------------------------------
# Identity
# ---------------------------------------------------------------------------

PROJECT_NAME = "tinyllm"
"""Short name used for checkpoint dirs, GGUF filenames and the served model id."""


@dataclass(frozen=True)
class HubConfig:
    """Hugging Face Hub coordinates.

    The Hub is this project's transport between tiers: Colab pushes, the laptop
    pulls. Set ``user`` once (or export HF_USER) and everything else derives.
    """

    user: str = "pythonstudentiam"  # your HF username
    model_repo_suffix: str = PROJECT_NAME
    ckpt_repo_suffix: str = f"{PROJECT_NAME}-checkpoints"

    @property
    def model_repo(self) -> str:
        """Final published model: safetensors + tokenizer + GGUF."""
        return f"{self.user}/{self.model_repo_suffix}"

    @property
    def ckpt_repo(self) -> str:
        """Mid-training checkpoints. Separate repo so the model repo stays clean.

        This exists because free-tier Colab reclaims runtimes without warning;
        checkpoints written only to Colab's local disk are not durable.
        """
        return f"{self.user}/{self.ckpt_repo_suffix}"


# ---------------------------------------------------------------------------
# Tokenizer
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class TokenizerConfig:
    """SentencePiece BPE settings.

    Why SentencePiece and not the `tokenizers` library: llama.cpp's
    convert_hf_to_gguf.py identifies BPE pre-tokenizers by hashing them against
    a hardcoded registry and raises NotImplementedError on anything unknown --
    which includes every custom `tokenizers` BPE. The SentencePiece path calls
    _set_vocab_sentencepiece() instead, which does no hash lookup at all.
    See docs/02-tokenizer.md.
    """

    vocab_size: int = 8192
    model_type: str = "bpe"
    character_coverage: float = 1.0  # 1.0 is right for pure-ASCII-ish English
    train_sentences: int = 400_000  # sample of the corpus used to fit the vocab
    max_sentence_length: int = 8192

    # SentencePiece assigns these ids in this order; llama.cpp expects them.
    unk_id: int = 0
    bos_id: int = 1
    eos_id: int = 2
    pad_id: int = 3

    unk_piece: str = "<unk>"
    bos_piece: str = "<s>"
    eos_piece: str = "</s>"
    pad_piece: str = "<pad>"

    # ChatML control tokens, added as user_defined_symbols so they tokenize
    # atomically (never split into subwords) and survive round-tripping.
    im_start: str = "<|im_start|>"
    im_end: str = "<|im_end|>"

    @property
    def user_defined_symbols(self) -> list[str]:
        return [self.im_start, self.im_end]

    @property
    def chat_template(self) -> str:
        """ChatML. Embedded into tokenizer_config.json, and from there into the
        GGUF metadata, which is how llama-server learns to format turns.
        """
        return (
            "{% for message in messages %}"
            "{{ '<|im_start|>' + message['role'] + '\\n' + message['content'] + '<|im_end|>' + '\\n' }}"
            "{% endfor %}"
            "{% if add_generation_prompt %}"
            "{{ '<|im_start|>assistant\\n' }}"
            "{% endif %}"
        )


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ModelConfig:
    """Llama-architecture decoder. ~15.7M parameters at these defaults.

    Sized so pretraining fits in ~45 min on a free Colab T4 and inference is
    instant on a 2-core CPU.
    """

    hidden_size: int = 384
    num_hidden_layers: int = 8
    num_attention_heads: int = 6
    num_key_value_heads: int = 2  # real GQA: 3 query heads share each KV head
    intermediate_size: int = 1024  # SwiGLU, ~2.67x hidden
    vocab_size: int = TokenizerConfig.vocab_size
    max_position_embeddings: int = 512
    rope_theta: float = 10_000.0
    rms_norm_eps: float = 1e-5
    tie_word_embeddings: bool = True  # saves 3.1M params -- 20% of the model
    attention_bias: bool = False
    mlp_bias: bool = False
    initializer_range: float = 0.02

    def __post_init__(self) -> None:
        if self.hidden_size % self.num_attention_heads:
            raise ValueError("hidden_size must be divisible by num_attention_heads")
        if self.num_attention_heads % self.num_key_value_heads:
            raise ValueError("num_attention_heads must be divisible by num_key_value_heads")

    @property
    def head_dim(self) -> int:
        return self.hidden_size // self.num_attention_heads

    @property
    def kv_dim(self) -> int:
        """Width of the K and V projections. Smaller than hidden_size under GQA."""
        return self.num_key_value_heads * self.head_dim

    @property
    def n_rep(self) -> int:
        """How many query heads share each KV head."""
        return self.num_attention_heads // self.num_key_value_heads

    def param_breakdown(self) -> dict[str, int]:
        """Exact parameter count by component. Verified against the real model
        in notebook 03 -- if these disagree, one of them has a bug.
        """
        h, i, v = self.hidden_size, self.intermediate_size, self.vocab_size
        embedding = v * h
        attn = 2 * h * h + 2 * h * self.kv_dim  # q,o are h*h; k,v are h*kv_dim
        mlp = 3 * h * i  # gate, up, down
        norms = 2 * h  # input_layernorm + post_attention_layernorm
        per_layer = attn + mlp + norms
        blocks = per_layer * self.num_hidden_layers
        final_norm = h
        lm_head = 0 if self.tie_word_embeddings else v * h

        return {
            "embedding": embedding,
            "attention": attn * self.num_hidden_layers,
            "mlp": mlp * self.num_hidden_layers,
            "layernorms": norms * self.num_hidden_layers + final_norm,
            "lm_head": lm_head,
            "per_layer": per_layer,
            "blocks_total": blocks,
            "non_embedding": blocks + final_norm + lm_head,
            "total": embedding + blocks + final_norm + lm_head,
        }

    @property
    def n_params(self) -> int:
        return self.param_breakdown()["total"]

    def flops_per_token(self, seq_len: int | None = None) -> float:
        """Forward+backward FLOPs per training token.

        The familiar 6N rule plus the attention term that 6N omits. At ctx 512
        attention is a small slice, but the term is here so notebook 04 can
        check measured throughput against a prediction that is actually right.
        """
        seq_len = seq_len or self.max_position_embeddings
        dense = 6 * self.param_breakdown()["non_embedding"]
        attn = 6 * 2 * self.num_hidden_layers * self.hidden_size * seq_len
        return dense + attn

    def to_hf_config(self) -> Any:
        """Build a transformers LlamaConfig. Lazy import -- torch/transformers
        are Colab-only and must never be required on the local tier.
        """
        from transformers import LlamaConfig

        return LlamaConfig(
            hidden_size=self.hidden_size,
            num_hidden_layers=self.num_hidden_layers,
            num_attention_heads=self.num_attention_heads,
            num_key_value_heads=self.num_key_value_heads,
            intermediate_size=self.intermediate_size,
            vocab_size=self.vocab_size,
            max_position_embeddings=self.max_position_embeddings,
            rope_theta=self.rope_theta,
            rms_norm_eps=self.rms_norm_eps,
            tie_word_embeddings=self.tie_word_embeddings,
            attention_bias=self.attention_bias,
            mlp_bias=self.mlp_bias,
            initializer_range=self.initializer_range,
            hidden_act="silu",
            bos_token_id=TokenizerConfig.bos_id,
            eos_token_id=TokenizerConfig.eos_id,
            pad_token_id=TokenizerConfig.pad_id,
        )


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class DataConfig:
    """TinyStories, streamed.

    The full dataset is ~7.6 GB. We stream it rather than downloading, which is
    both how you handle corpora larger than local disk and how you avoid
    exhausting Colab's disk quota.
    """

    dataset_id: str = "roneneldan/TinyStories"
    instruct_dataset_id: str = "roneneldan/TinyStoriesInstruct"
    train_split: str = "train"
    val_split: str = "validation"

    seq_len: int = ModelConfig.max_position_embeddings
    val_tokens: int = 1_000_000  # held-out budget for perplexity
    shard_tokens: int = 25_000_000  # tokens per .bin shard written during prep
    seed: int = 1337


# ---------------------------------------------------------------------------
# Pretraining
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class TrainConfig:
    """~327M tokens, roughly Chinchilla-optimal (20x params) for a 15.7M model."""

    # Batch: micro_batch x grad_accum x seq_len = tokens per optimizer step
    micro_batch_size: int = 32
    grad_accum_steps: int = 4

    # 2,500 steps = ~164M tokens = ~10 tokens/parameter, about half of the
    # Chinchilla-optimal ~20. A deliberate trade: free-tier Colab GPU hours are
    # the binding constraint, and this halves them. The model comes out
    # measurably weaker but still fluent on TinyStories, and every downstream
    # stage is unaffected. Raise back to 5_000 when GPU time allows -- resuming
    # from a checkpoint written at the lower budget works fine.
    max_steps: int = 2_500

    learning_rate: float = 6e-4
    min_lr_ratio: float = 0.1  # cosine floor = lr * ratio
    warmup_steps: int = 200
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95
    grad_clip: float = 1.0

    # T4 is Turing (SM75): no bf16. fp16 AMP + GradScaler is the only option.
    dtype: str = "fp16"
    compile_model: bool = False  # torch.compile is a net loss at this size

    eval_every: int = 250
    eval_batches: int = 40
    sample_every: int = 500
    log_every: int = 10
    checkpoint_every: int = 500  # also the Hub push cadence
    keep_last_n_checkpoints: int = 2

    seed: int = 1337

    # A cheap end-to-end rehearsal. Run this before committing 45 minutes:
    # it exercises every code path in the training loop at 1/100th the cost.
    smoke_max_steps: int = 50
    smoke_stories: int = 2_000

    @property
    def tokens_per_step(self) -> int:
        return self.micro_batch_size * self.grad_accum_steps * DataConfig.seq_len

    @property
    def total_tokens(self) -> int:
        return self.tokens_per_step * self.max_steps

    @property
    def min_lr(self) -> float:
        return self.learning_rate * self.min_lr_ratio

    def lr_at(self, step: int) -> float:
        """Linear warmup then cosine decay. Mirrored exactly in train.py."""
        if step < self.warmup_steps:
            return self.learning_rate * (step + 1) / self.warmup_steps
        if step >= self.max_steps:
            return self.min_lr
        progress = (step - self.warmup_steps) / max(1, self.max_steps - self.warmup_steps)
        coeff = 0.5 * (1.0 + math.cos(math.pi * progress))
        return self.min_lr + coeff * (self.learning_rate - self.min_lr)

    def chinchilla_ratio(self, model: ModelConfig) -> float:
        """Tokens per parameter. ~20 is the Chinchilla-optimal neighbourhood."""
        return self.total_tokens / model.n_params


# ---------------------------------------------------------------------------
# Instruction tuning
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class SFTConfig:
    """Turns the text-completer into something that responds to instructions."""

    micro_batch_size: int = 16
    grad_accum_steps: int = 4
    max_steps: int = 1_500
    learning_rate: float = 1e-4  # ~6x below pretrain: don't wreck what was learned
    min_lr_ratio: float = 0.1
    warmup_steps: int = 50
    weight_decay: float = 0.0
    # Same names as TrainConfig so train.make_optimizer() accepts either config.
    beta1: float = 0.9
    beta2: float = 0.95
    grad_clip: float = 1.0
    seq_len: int = 512

    # Prompt tokens are set to this so loss is computed on completions only.
    # Getting this wrong fails silently -- the model still trains, just worse.
    # Notebook 06 decodes a batch and asserts the mask is where it should be.
    ignore_index: int = -100

    eval_every: int = 200
    log_every: int = 10
    checkpoint_every: int = 500
    seed: int = 1337


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class GenConfig:
    max_new_tokens: int = 256
    temperature: float = 0.8
    top_p: float = 0.95
    top_k: int = 40
    repetition_penalty: float = 1.1

    eval_prompts: tuple[str, ...] = (
        "Once upon a time, there was a little girl named Lily.",
        "Tom and Sara went to the park. They saw a big",
        "The cat was very hungry, so it",
    )
    eval_instructions: tuple[str, ...] = (
        "Write a story about a lost puppy who finds its way home.",
        "Write a short story using the words: ball, tree, happy.",
        "Tell me a story about a brave little boat.",
    )


# ---------------------------------------------------------------------------
# Quantization (local tier)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class QuantConfig:
    """Quantization levels compared in notebook 10.

    Expect Q4_K_M to degrade noticeably here. Quantization advice is written
    for 7B+ models; a 15.7M model has far less parameter redundancy to absorb
    rounding error. That contrast is the point of the experiment.
    """

    levels: tuple[str, ...] = ("Q8_0", "Q5_K_M", "Q4_K_M")
    perplexity_ctx: int = 512
    perplexity_chunks: int = 40


# ---------------------------------------------------------------------------
# Local runtime
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ServeConfig:
    """llama-server on this laptop."""

    host: str = "127.0.0.1"
    port: int = 8080
    threads: int = 4  # i5-6200U: 2 physical cores, 4 logical
    ctx_size: int = 512  # must not exceed ModelConfig.max_position_embeddings
    served_model_name: str = PROJECT_NAME
    default_quant: str = "Q8_0"

    # Pinned so a llama.cpp release churn never silently breaks setup_local.ps1.
    llamacpp_build: str = "b10107"
    llamacpp_asset: str = "llama-b10107-bin-win-cpu-x64.zip"

    @property
    def base_url(self) -> str:
        return f"http://{self.host}:{self.port}/v1"

    @property
    def llamacpp_url(self) -> str:
        return (
            "https://github.com/ggml-org/llama.cpp/releases/download/"
            f"{self.llamacpp_build}/{self.llamacpp_asset}"
        )


# ---------------------------------------------------------------------------
# Instances -- import these, don't re-instantiate
# ---------------------------------------------------------------------------

hub = HubConfig()
tok_cfg = TokenizerConfig()
model_cfg = ModelConfig()
data_cfg = DataConfig()
train_cfg = TrainConfig()
sft_cfg = SFTConfig()
gen_cfg = GenConfig()
quant_cfg = QuantConfig()
serve_cfg = ServeConfig()


def summary() -> str:
    """Human-readable snapshot. Printed at the top of every notebook so each
    run's artifacts carry a record of the config that produced them.
    """
    pb = model_cfg.param_breakdown()
    lines = [
        f"{'=' * 62}",
        f" {PROJECT_NAME}  --  end-to-end LLM lifecycle",
        f"{'=' * 62}",
        "",
        " MODEL",
        f"   hidden={model_cfg.hidden_size}  layers={model_cfg.num_hidden_layers}  "
        f"q_heads={model_cfg.num_attention_heads}  kv_heads={model_cfg.num_key_value_heads}"
        f"  head_dim={model_cfg.head_dim}",
        f"   intermediate={model_cfg.intermediate_size}  vocab={model_cfg.vocab_size}"
        f"  ctx={model_cfg.max_position_embeddings}  tied={model_cfg.tie_word_embeddings}",
        f"   GQA: {model_cfg.n_rep} query heads per KV head "
        f"({100 * (1 - model_cfg.kv_dim / model_cfg.hidden_size):.0f}% smaller KV cache)",
        "",
        " PARAMETERS",
        f"   embedding      {pb['embedding']:>12,}",
        f"   attention      {pb['attention']:>12,}",
        f"   mlp            {pb['mlp']:>12,}",
        f"   layernorms     {pb['layernorms']:>12,}",
        f"   lm_head        {pb['lm_head']:>12,}  (tied -> free)",
        f"   {'-' * 28}",
        f"   total          {pb['total']:>12,}  ({pb['total'] / 1e6:.2f}M)",
        f"   non-embedding  {pb['non_embedding']:>12,}",
        "",
        " PRETRAIN",
        f"   tokens/step={train_cfg.tokens_per_step:,}  steps={train_cfg.max_steps:,}"
        f"  total={train_cfg.total_tokens / 1e6:.0f}M tokens",
        f"   tokens/param={train_cfg.chinchilla_ratio(model_cfg):.1f}  (Chinchilla-optimal ~20)",
        f"   lr={train_cfg.learning_rate}  warmup={train_cfg.warmup_steps}  dtype={train_cfg.dtype}",
        f"   fwd+bwd FLOPs/token={model_cfg.flops_per_token():.3e}",
        f"   total train FLOPs={model_cfg.flops_per_token() * train_cfg.total_tokens:.3e}",
        "",
        " HUB",
        f"   model={hub.model_repo}",
        f"   ckpts={hub.ckpt_repo}",
        f"{'=' * 62}",
    ]
    return "\n".join(lines)


def as_dict() -> dict[str, Any]:
    """Full config as plain data -- saved alongside checkpoints and the model
    card so any artifact can be traced back to the settings that made it.

    This is also the bridge the PowerShell scripts read (see
    scripts/_common.ps1 :: Get-TinyllmConfig). ``dataclasses.asdict`` serializes
    *fields only*, so computed properties like ``hub.model_repo`` would silently
    come back empty on the PowerShell side -- and a guard that reads an empty
    string does not fire. They are merged in explicitly below.
    """
    return {
        "project": PROJECT_NAME,
        "hub": {
            **asdict(hub),
            "model_repo": hub.model_repo,
            "ckpt_repo": hub.ckpt_repo,
        },
        "tokenizer": {
            **asdict(tok_cfg),
            "user_defined_symbols": tok_cfg.user_defined_symbols,
            "chat_template": tok_cfg.chat_template,
        },
        "model": {
            **asdict(model_cfg),
            "head_dim": model_cfg.head_dim,
            "kv_dim": model_cfg.kv_dim,
            "n_rep": model_cfg.n_rep,
            "n_params": model_cfg.n_params,
        },
        "data": asdict(data_cfg),
        "train": {
            **asdict(train_cfg),
            "tokens_per_step": train_cfg.tokens_per_step,
            "total_tokens": train_cfg.total_tokens,
            "min_lr": train_cfg.min_lr,
        },
        "sft": asdict(sft_cfg),
        "gen": asdict(gen_cfg),
        "quant": asdict(quant_cfg),
        "serve": {
            **asdict(serve_cfg),
            "base_url": serve_cfg.base_url,
            "llamacpp_url": serve_cfg.llamacpp_url,
        },
        "derived": {
            "head_dim": model_cfg.head_dim,
            "kv_dim": model_cfg.kv_dim,
            "n_params": model_cfg.n_params,
            "param_breakdown": model_cfg.param_breakdown(),
            "tokens_per_step": train_cfg.tokens_per_step,
            "total_tokens": train_cfg.total_tokens,
            "flops_per_token": model_cfg.flops_per_token(),
        },
    }


if __name__ == "__main__":
    print(summary())
