"""Stage 3 -- the Llama decoder, written from scratch to be read.

This is the reference implementation. It is *not* what gets trained: stage 4
trains ``transformers.LlamaForCausalLM`` instead, because that model's tensor
names are exactly what ``convert_hf_to_gguf.py`` knows how to read, and
inheriting conversion risk to save a dependency would be a bad trade.

What this file buys you is understanding you can verify. Module and parameter
names here match HuggingFace's byte for byte, so a state dict loads into either
model unchanged -- which is what lets ``parity.py`` prove the two are the same
function rather than merely asserting they look similar.

The four things that make this "Llama" rather than "GPT":

  RMSNorm    no mean subtraction, no bias -- just scale by RMS. Cheaper than
             LayerNorm and empirically just as good.
  RoPE       position encoded by *rotating* q/k in 2D subspaces, so attention
             scores depend on relative distance and nothing is ever added to
             the residual stream.
  GQA        query heads outnumber key/value heads. Here 6 query heads share
             2 KV heads, cutting the KV cache by 67% at negligible quality cost.
  SwiGLU     a gated MLP: ``down(silu(gate(x)) * up(x))``. Three matrices
             instead of two, so the hidden width is ~2.67x rather than 4x.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from tinyllm.config import ModelConfig, model_cfg


# ---------------------------------------------------------------------------
# Normalization
# ---------------------------------------------------------------------------

class RMSNorm(nn.Module):
    """Root-mean-square layer norm.

    Note the explicit float32 cast: the variance of a half-precision activation
    can overflow or lose all its precision, and normalizing in fp16 is a classic
    source of silent training divergence. Compute in fp32, scale, cast back.
    """

    def __init__(self, hidden_size: int, eps: float = 1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        x = x.to(torch.float32)
        variance = x.pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * x.to(dtype)


# ---------------------------------------------------------------------------
# Rotary position embeddings
# ---------------------------------------------------------------------------

class RotaryEmbedding(nn.Module):
    """Precomputes cos/sin tables for RoPE.

    Frequencies are geometrically spaced: ``1 / theta^(2i/d)``. Low-index pairs
    rotate fast (they encode fine local position), high-index pairs rotate
    slowly (coarse global position). Because the rotation applied to a query at
    position *m* and a key at position *n* composes to a rotation by *m - n*,
    the resulting attention score depends only on relative distance.
    """

    def __init__(self, dim: int, max_seq_len: int, theta: float = 10_000.0):
        super().__init__()
        inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

        t = torch.arange(max_seq_len, dtype=torch.float32)
        freqs = torch.outer(t, inv_freq)              # (seq, dim/2)
        emb = torch.cat((freqs, freqs), dim=-1)       # (seq, dim)
        self.register_buffer("cos_cached", emb.cos(), persistent=False)
        self.register_buffer("sin_cached", emb.sin(), persistent=False)

    def forward(self, seq_len: int, offset: int = 0, dtype=torch.float32):
        """``offset`` is where this chunk starts -- non-zero during cached
        generation, when we feed one token at absolute position `offset`."""
        cos = self.cos_cached[offset : offset + seq_len].to(dtype)
        sin = self.sin_cached[offset : offset + seq_len].to(dtype)
        return cos, sin


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Treat the head dim as two halves and rotate them into each other.

    HuggingFace pairs dimension *i* with *i + d/2* (not *i* with *i+1*, which
    the RoPE paper's notation suggests). The two conventions are equivalent up
    to a permutation of the head dimension, but they are NOT interchangeable
    for a given set of weights -- matching HF here is what makes parity work.
    """
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor):
    """Rotate queries and keys. cos/sin broadcast over batch and head dims."""
    cos = cos.unsqueeze(0).unsqueeze(0)   # (1, 1, seq, head_dim)
    sin = sin.unsqueeze(0).unsqueeze(0)
    return (q * cos) + (rotate_half(q) * sin), (k * cos) + (rotate_half(k) * sin)


def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """Expand KV heads to match query heads for grouped-query attention.

    This is a view-and-reshape, so it costs no FLOPs. The saving GQA buys is in
    the *cache*: only ``num_key_value_heads`` worth of K and V are ever stored,
    which is what makes long-context inference affordable.
    """
    b, n_kv, s, d = x.shape
    if n_rep == 1:
        return x
    x = x[:, :, None, :, :].expand(b, n_kv, n_rep, s, d)
    return x.reshape(b, n_kv * n_rep, s, d)


# ---------------------------------------------------------------------------
# Attention
# ---------------------------------------------------------------------------

class Attention(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.num_heads = cfg.num_attention_heads
        self.num_kv_heads = cfg.num_key_value_heads
        self.head_dim = cfg.head_dim
        self.n_rep = cfg.n_rep

        # No biases: Llama drops them everywhere. They cost parameters and
        # buy nothing once you have normalization layers.
        self.q_proj = nn.Linear(cfg.hidden_size, self.num_heads * self.head_dim, bias=cfg.attention_bias)
        self.k_proj = nn.Linear(cfg.hidden_size, self.num_kv_heads * self.head_dim, bias=cfg.attention_bias)
        self.v_proj = nn.Linear(cfg.hidden_size, self.num_kv_heads * self.head_dim, bias=cfg.attention_bias)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, cfg.hidden_size, bias=cfg.attention_bias)

    def forward(self, x, cos, sin, attn_mask=None, kv_cache=None, use_cache=False):
        b, s, _ = x.shape

        q = self.q_proj(x).view(b, s, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(b, s, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(b, s, self.num_kv_heads, self.head_dim).transpose(1, 2)

        q, k = apply_rotary_pos_emb(q, k, cos, sin)

        # Append to the cache *after* rotating: cached keys already carry their
        # own positions, so re-rotating them would corrupt the encoding.
        #
        # `use_cache` is a separate flag from `kv_cache` on purpose. Conflating
        # them is a real bug this code once had: generate() seeds the cache list
        # with None entries, so "kv_cache is None" is true on the *first* step of
        # a cached run as well as on an uncached run. Keying the write off that
        # meant the cache was never populated, and every step after the first
        # attended only to its own token.
        if kv_cache is not None:
            past_k, past_v = kv_cache
            if past_k is not None:
                k = torch.cat([past_k, k], dim=2)
                v = torch.cat([past_v, v], dim=2)

        new_cache = (k, v) if use_cache else None

        k = repeat_kv(k, self.n_rep)
        v = repeat_kv(v, self.n_rep)

        # Written out eagerly rather than via scaled_dot_product_attention.
        # This model is a reference, so legibility beats throughput; the model
        # that actually trains is HuggingFace's.
        scores = torch.matmul(q, k.transpose(2, 3)) / math.sqrt(self.head_dim)
        if attn_mask is not None:
            scores = scores + attn_mask
        scores = F.softmax(scores.float(), dim=-1).to(q.dtype)

        out = torch.matmul(scores, v)
        out = out.transpose(1, 2).contiguous().view(b, s, -1)
        return self.o_proj(out), new_cache


# ---------------------------------------------------------------------------
# MLP
# ---------------------------------------------------------------------------

class MLP(nn.Module):
    """SwiGLU. ``gate`` decides how much of ``up`` survives, per channel."""

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.gate_proj = nn.Linear(cfg.hidden_size, cfg.intermediate_size, bias=cfg.mlp_bias)
        self.up_proj = nn.Linear(cfg.hidden_size, cfg.intermediate_size, bias=cfg.mlp_bias)
        self.down_proj = nn.Linear(cfg.intermediate_size, cfg.hidden_size, bias=cfg.mlp_bias)

    def forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


# ---------------------------------------------------------------------------
# Block
# ---------------------------------------------------------------------------

class DecoderLayer(nn.Module):
    """Pre-norm residual block.

    Pre-norm (normalize *going into* each sublayer) rather than post-norm means
    the residual stream is never renormalized, so gradients reach layer 0
    without attenuation. It is why deep transformers train without warmup
    tricks that post-norm models needed.
    """

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.self_attn = Attention(cfg)
        self.mlp = MLP(cfg)
        self.input_layernorm = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps)

    def forward(self, x, cos, sin, attn_mask=None, kv_cache=None, use_cache=False):
        h, new_cache = self.self_attn(
            self.input_layernorm(x), cos, sin, attn_mask, kv_cache, use_cache
        )
        x = x + h
        x = x + self.mlp(self.post_attention_layernorm(x))
        return x, new_cache


# ---------------------------------------------------------------------------
# Full model
# ---------------------------------------------------------------------------

class TinyLlamaModel(nn.Module):
    """Transformer trunk. Named ``model.*`` to mirror HF's nesting."""

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.embed_tokens = nn.Embedding(cfg.vocab_size, cfg.hidden_size)
        self.layers = nn.ModuleList([DecoderLayer(cfg) for _ in range(cfg.num_hidden_layers)])
        self.norm = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps)
        self.rotary = RotaryEmbedding(cfg.head_dim, cfg.max_position_embeddings, cfg.rope_theta)

    def forward(self, input_ids, kv_caches=None, offset: int = 0):
        # Passing a list of caches -- even a list of Nones -- is the request to
        # cache. See the note in Attention.forward.
        use_cache = kv_caches is not None

        b, s = input_ids.shape
        x = self.embed_tokens(input_ids)
        cos, sin = self.rotary(s, offset=offset, dtype=x.dtype)

        # Causal mask. During cached generation s == 1 and every cached position
        # is visible, so no mask is needed at all.
        attn_mask = None
        if s > 1:
            total = s + offset
            mask = torch.full((s, total), float("-inf"), device=x.device, dtype=x.dtype)
            mask = torch.triu(mask, diagonal=1 + offset)
            attn_mask = mask[None, None, :, :]

        new_caches = []
        for i, layer in enumerate(self.layers):
            cache = kv_caches[i] if kv_caches is not None else None
            x, nc = layer(x, cos, sin, attn_mask, cache, use_cache)
            new_caches.append(nc)

        return self.norm(x), new_caches


class TinyLlamaForCausalLM(nn.Module):
    """Trunk plus the language-modelling head.

    Parameter names match ``transformers.LlamaForCausalLM`` exactly, so::

        scratch.load_state_dict(hf_model.state_dict())

    just works. That interchangeability is the whole point -- see parity.py.
    """

    def __init__(self, cfg: ModelConfig = model_cfg):
        super().__init__()
        self.cfg = cfg
        self.model = TinyLlamaModel(cfg)
        self.lm_head = nn.Linear(cfg.hidden_size, cfg.vocab_size, bias=False)

        if cfg.tie_word_embeddings:
            # One tensor, two uses: 3.1M parameters saved, 20% of the model.
            # Sound at this scale because the embedding and the unembedding are
            # both maps between the same two spaces.
            self.lm_head.weight = self.model.embed_tokens.weight

        self.apply(self._init_weights)

    def _init_weights(self, module):
        std = self.cfg.initializer_range
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=std)

    def forward(self, input_ids, labels=None, kv_caches=None, offset: int = 0):
        hidden, new_caches = self.model(input_ids, kv_caches=kv_caches, offset=offset)
        logits = self.lm_head(hidden)

        loss = None
        if labels is not None:
            # Shift so position t predicts token t+1.
            loss = F.cross_entropy(
                logits[:, :-1, :].reshape(-1, logits.size(-1)).float(),
                labels[:, 1:].reshape(-1),
                ignore_index=-100,
            )
        return logits, loss, new_caches

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 128,
        temperature: float = 0.8,
        top_k: int | None = 40,
        eos_token_id: int | None = None,
        use_cache: bool = True,
    ) -> torch.Tensor:
        """Autoregressive sampling.

        With ``use_cache`` the first pass processes the whole prompt and every
        later pass processes a single token, reusing stored K/V. The alternative
        -- re-running the full prefix each step -- is quadratic and is only kept
        as an option because parity.py uses it to prove the cache is correct.
        """
        self.eval()
        caches = [None] * self.cfg.num_hidden_layers if use_cache else None
        offset = 0
        cur = input_ids

        for _ in range(max_new_tokens):
            if use_cache:
                logits, _, caches = self.forward(cur, kv_caches=caches, offset=offset)
                offset += cur.shape[1]
            else:
                window = input_ids[:, -self.cfg.max_position_embeddings :]
                logits, _, _ = self.forward(window)

            next_logits = logits[:, -1, :]
            if temperature <= 0:
                next_token = next_logits.argmax(dim=-1, keepdim=True)
            else:
                next_logits = next_logits / temperature
                if top_k is not None:
                    kth = torch.topk(next_logits, min(top_k, next_logits.size(-1)))[0][..., -1:]
                    next_logits = next_logits.masked_fill(next_logits < kth, float("-inf"))
                probs = F.softmax(next_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)

            input_ids = torch.cat([input_ids, next_token], dim=1)
            cur = next_token if use_cache else input_ids

            if eos_token_id is not None and (next_token == eos_token_id).all():
                break

            if input_ids.shape[1] >= self.cfg.max_position_embeddings:
                break

        return input_ids

    def num_parameters(self, non_embedding: bool = False) -> int:
        n = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n -= self.model.embed_tokens.weight.numel()
        return n
