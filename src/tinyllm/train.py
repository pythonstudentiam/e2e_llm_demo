"""Stage 4 -- pretraining.

The loop itself is unremarkable; the operational details are the point.

**fp16, not bf16.** A Colab T4 is Turing (SM75), which has no bf16 support. That
forces fp16 autocast plus a ``GradScaler``: fp16 gradients underflow to zero
without loss scaling. bf16 would let us skip the scaler entirely -- on an A100
or newer, do that.

**Durable checkpoints.** Free-tier Colab reclaims runtimes without warning. A
checkpoint on the runtime's local disk is not a checkpoint, so state is pushed
to a Hub repo. Resume restores the optimizer moments, the scaler scale, *and*
the data sampler's RNG state -- restoring only the weights silently restarts the
data order and quietly re-trains on tokens the model already saw.

**Decoupled weight decay.** Decay applies to matrices only. Applying it to
RMSNorm gains and biases pulls them toward zero, which is not regularization,
just damage.
"""

from __future__ import annotations

import csv
import json
import math
import time
from pathlib import Path

import numpy as np
import torch

from tinyllm import config as cfgmod
from tinyllm.config import ModelConfig, TrainConfig, data_cfg, hub, model_cfg, train_cfg
from tinyllm.data import causal_loss, get_batch, iter_eval_batches

# T4 fp16 tensor-core peak. Used only for the MFU readout.
T4_PEAK_FLOPS = 65e12


# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------

def build_model(cfg: ModelConfig = model_cfg, device: str = "cuda"):
    """Instantiate the model that actually trains.

    HuggingFace's implementation, not ours, because its tensor names are what
    convert_hf_to_gguf.py reads in stage 8. parity.py has already established
    the two compute the same function.
    """
    from transformers import LlamaForCausalLM

    hf_cfg = cfg.to_hf_config()
    hf_cfg._attn_implementation = "sdpa"  # fused attention; ~1.3x on a T4
    model = LlamaForCausalLM(hf_cfg).to(device)
    return model


def make_optimizer(model, cfg: TrainConfig = train_cfg):
    """AdamW with decay on matrices only.

    ``fused=True`` keeps the optimizer step on-GPU as a single kernel. At this
    model size the step is otherwise a surprisingly large slice of wall time.
    """
    decay, no_decay = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        (decay if p.dim() >= 2 else no_decay).append(p)

    groups = [
        {"params": decay, "weight_decay": cfg.weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]

    use_fused = torch.cuda.is_available()
    opt = torch.optim.AdamW(
        groups,
        lr=cfg.learning_rate,
        betas=(cfg.beta1, cfg.beta2),
        eps=1e-8,
        fused=use_fused,
    )
    return opt, {"decay_tensors": len(decay), "no_decay_tensors": len(no_decay)}


# ---------------------------------------------------------------------------
# Checkpointing
# ---------------------------------------------------------------------------

def save_checkpoint(path: Path, model, optimizer, scaler, step: int, rng: np.random.Generator, metrics: dict):
    """Everything needed to make a resume bit-identical to an uninterrupted run."""
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "step": step,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scaler": scaler.state_dict(),
            "np_rng_state": rng.bit_generator.state,
            "torch_rng_state": torch.get_rng_state(),
            "cuda_rng_state": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
            "metrics": metrics,
            "config": cfgmod.as_dict(),
        },
        path,
    )
    return path


def load_checkpoint(path: Path, model, optimizer=None, scaler=None, rng: np.random.Generator | None = None):
    """Restore a run. Returns the step to resume from."""
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model"])
    if optimizer is not None and ckpt.get("optimizer"):
        optimizer.load_state_dict(ckpt["optimizer"])
    if scaler is not None and ckpt.get("scaler"):
        scaler.load_state_dict(ckpt["scaler"])
    if rng is not None and ckpt.get("np_rng_state"):
        rng.bit_generator.state = ckpt["np_rng_state"]
    if ckpt.get("torch_rng_state") is not None:
        torch.set_rng_state(ckpt["torch_rng_state"].cpu().to(torch.uint8))
    if ckpt.get("cuda_rng_state") is not None and torch.cuda.is_available():
        try:
            torch.cuda.set_rng_state_all(ckpt["cuda_rng_state"])
        except Exception:
            pass  # different GPU count on resume; harmless
    return ckpt["step"], ckpt.get("metrics", {})


def push_checkpoint(path: Path, repo_id: str = None, filename: str = "latest.pt"):
    """Upload to the Hub, overwriting the previous one.

    A single rolling file rather than one per step: the aim is surviving a
    disconnect, not keeping training history, and 190 MB per checkpoint adds up.
    """
    from huggingface_hub import HfApi

    repo_id = repo_id or hub.ckpt_repo
    api = HfApi()
    api.create_repo(repo_id, repo_type="model", exist_ok=True, private=True)
    api.upload_file(
        path_or_fileobj=str(path),
        path_in_repo=filename,
        repo_id=repo_id,
        repo_type="model",
    )
    return f"{repo_id}/{filename}"


def pull_checkpoint(repo_id: str = None, filename: str = "latest.pt", local_dir: Path = Path("checkpoints")):
    """Fetch the rolling checkpoint, or return None if there isn't one yet."""
    from huggingface_hub import hf_hub_download
    from huggingface_hub.errors import EntryNotFoundError, RepositoryNotFoundError

    repo_id = repo_id or hub.ckpt_repo
    try:
        return Path(hf_hub_download(repo_id=repo_id, filename=filename, local_dir=str(local_dir)))
    except (EntryNotFoundError, RepositoryNotFoundError, OSError):
        return None


# ---------------------------------------------------------------------------
# Evaluation and sampling during training
# ---------------------------------------------------------------------------

@torch.no_grad()
def estimate_loss(model, tokens, batch_size: int, seq_len: int, n_batches: int, device: str = "cuda") -> float:
    """Mean loss over a fixed, deterministic set of validation windows."""
    model.eval()
    losses = []
    for x, y in iter_eval_batches(tokens, batch_size, seq_len, n_batches, device=device):
        with torch.autocast("cuda", dtype=torch.float16, enabled=device.startswith("cuda")):
            logits = model(input_ids=x).logits
        losses.append(causal_loss(logits, y).item())
    model.train()
    return float(np.mean(losses))


@torch.no_grad()
def assert_loss_convention(model, x, y, tol: float = 1e-3) -> dict:
    """Guard against the double-shift bug, which is otherwise invisible.

    Passing ``labels=y`` to a HuggingFace causal LM double-shifts the targets and
    trains the model to predict two tokens ahead. It converges to a plausible
    loss curve and produces incoherent text -- a full training run's worth of
    wasted compute before anything looks wrong.

    On identical positions, HF's internal shift with ``labels=x`` must agree with
    our explicit loss against ``y``. The double-shifted value is returned too, so
    the difference is visible rather than inferred.
    """
    model.eval()
    logits = model(input_ids=x).logits
    ours = causal_loss(logits[:, :-1, :], y[:, :-1])       # drop last to match HF
    hf = model(input_ids=x, labels=x).loss                  # HF shifts internally
    wrong = model(input_ids=x, labels=y).loss               # the bug
    model.train()

    delta = abs(ours.item() - hf.item())
    if delta > tol:
        raise AssertionError(
            f"Loss convention mismatch: explicit={ours.item():.6f} vs "
            f"HF labels=x {hf.item():.6f} (delta {delta:.2e}). "
            "get_batch's shift contract and the loss no longer agree."
        )
    return {
        "next_token_loss": ours.item(),
        "hf_labels_x": hf.item(),
        "double_shifted": wrong.item(),
        "delta": delta,
    }


@torch.no_grad()
def sample(model, sp, prompt: str, max_new_tokens: int = 100, temperature: float = 0.8, device: str = "cuda") -> str:
    """Generate a continuation. Watching these evolve is the most informative
    diagnostic in the whole run -- loss curves hide qualitative jumps."""
    model.eval()
    ids = [sp.bos_id()] + sp.EncodeAsIds(prompt)
    x = torch.tensor([ids], dtype=torch.long, device=device)
    out = model.generate(
        x,
        max_new_tokens=max_new_tokens,
        do_sample=temperature > 0,
        temperature=max(temperature, 1e-5),
        top_k=40,
        pad_token_id=sp.pad_id(),
        eos_token_id=sp.eos_id(),
    )
    model.train()
    return sp.DecodeIds(out[0].tolist())


# ---------------------------------------------------------------------------
# The loop
# ---------------------------------------------------------------------------

def train(
    train_tokens: np.ndarray,
    val_tokens: np.ndarray,
    sp=None,
    cfg: TrainConfig = train_cfg,
    mcfg: ModelConfig = model_cfg,
    out_dir: Path = Path("checkpoints"),
    device: str = "cuda",
    resume: bool = True,
    push_to_hub: bool = True,
    max_steps: int | None = None,
    log_path: Path | None = None,
):
    """Pretrain. Returns (model, history).

    ``max_steps`` overrides the config, which is how the smoke run reuses this
    exact code path rather than a simplified copy of it. The smoke run is only
    useful if it exercises the same code.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_path or (out_dir / "metrics.csv")
    total_steps = max_steps or cfg.max_steps

    torch.manual_seed(cfg.seed)
    rng = np.random.default_rng(cfg.seed)

    model = build_model(mcfg, device)
    optimizer, pg = make_optimizer(model, cfg)
    scaler = torch.amp.GradScaler("cuda", enabled=device.startswith("cuda"))

    start_step = 0
    history: list[dict] = []

    if resume:
        local = out_dir / "latest.pt"
        ckpt_path = local if local.exists() else pull_checkpoint(local_dir=out_dir)
        if ckpt_path and Path(ckpt_path).exists():
            start_step, prev = load_checkpoint(Path(ckpt_path), model, optimizer, scaler, rng)
            history = prev.get("history", [])
            print(f"  resumed from step {start_step:,}")

    model.train()

    # Verify the loss convention before spending any GPU time on it.
    _x, _y = get_batch(train_tokens, 4, data_cfg.seq_len, device=device, rng=np.random.default_rng(0))
    conv = assert_loss_convention(model, _x, _y)
    print(f"  loss check: next-token {conv['next_token_loss']:.4f} "
          f"(double-shifted would be {conv['double_shifted']:.4f})")
    del _x, _y

    n_params = sum(p.numel() for p in model.parameters())
    flops_per_token = mcfg.flops_per_token()
    print(f"  model: {n_params:,} params | {pg['decay_tensors']} decayed / {pg['no_decay_tensors']} not")
    print(f"  budget: {total_steps:,} steps x {cfg.tokens_per_step:,} tok = "
          f"{total_steps * cfg.tokens_per_step / 1e6:.0f}M tokens")

    new_file = not log_path.exists()
    logf = log_path.open("a", newline="", encoding="utf-8")
    writer = csv.writer(logf)
    if new_file:
        writer.writerow(["step", "train_loss", "val_loss", "lr", "grad_norm", "tokens_per_sec", "mfu", "elapsed_s"])

    t_start = time.time()
    t_last = t_start

    try:
        for step in range(start_step, total_steps):
            lr = cfg.lr_at(step)
            for g in optimizer.param_groups:
                g["lr"] = lr

            # ---- gradient accumulation ------------------------------------
            optimizer.zero_grad(set_to_none=True)
            accum_loss = 0.0
            for _ in range(cfg.grad_accum_steps):
                x, y = get_batch(train_tokens, cfg.micro_batch_size, data_cfg.seq_len, device=device, rng=rng)
                with torch.autocast("cuda", dtype=torch.float16, enabled=device.startswith("cuda")):
                    # NOT model(input_ids=x, labels=y): HF shifts labels
                    # internally and y is already shifted. See data.causal_loss.
                    logits = model(input_ids=x).logits
                    # Scale so the accumulated gradient equals the gradient of
                    # the mean loss over the full effective batch.
                    loss = causal_loss(logits, y) / cfg.grad_accum_steps
                scaler.scale(loss).backward()
                accum_loss += loss.item()

            # Unscale before clipping, or the clip threshold is applied to
            # gradients still multiplied by the loss scale and does nothing.
            scaler.unscale_(optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip).item()

            scaler.step(optimizer)
            scaler.update()

            # ---- logging ---------------------------------------------------
            if (step + 1) % cfg.log_every == 0 or step == start_step:
                now = time.time()
                dt = now - t_last
                t_last = now
                steps_done = cfg.log_every if (step + 1) % cfg.log_every == 0 else 1
                tps = cfg.tokens_per_step * steps_done / max(dt, 1e-6)
                mfu = (flops_per_token * tps) / T4_PEAK_FLOPS
                row = {
                    "step": step + 1, "train_loss": accum_loss, "val_loss": "",
                    "lr": lr, "grad_norm": grad_norm, "tokens_per_sec": tps,
                    "mfu": mfu, "elapsed_s": now - t_start,
                }
                history.append(row)
                writer.writerow(list(row.values()))
                logf.flush()
                print(f"  step {step + 1:>6,}/{total_steps:,} | loss {accum_loss:.4f} | "
                      f"lr {lr:.2e} | gn {grad_norm:5.2f} | {tps / 1e3:6.1f}k tok/s | mfu {mfu * 100:4.1f}%")

            # ---- validation ------------------------------------------------
            if (step + 1) % cfg.eval_every == 0 or (step + 1) == total_steps:
                vl = estimate_loss(model, val_tokens, cfg.micro_batch_size, data_cfg.seq_len,
                                   cfg.eval_batches, device=device)
                row = {
                    "step": step + 1, "train_loss": accum_loss, "val_loss": vl, "lr": lr,
                    "grad_norm": grad_norm, "tokens_per_sec": "", "mfu": "",
                    "elapsed_s": time.time() - t_start,
                }
                history.append(row)
                writer.writerow(list(row.values()))
                logf.flush()
                print(f"  --> val loss {vl:.4f} | ppl {math.exp(min(vl, 20)):.2f}")

            # ---- qualitative check -----------------------------------------
            if sp is not None and ((step + 1) % cfg.sample_every == 0):
                text = sample(model, sp, "Once upon a time", max_new_tokens=80, device=device)
                print(f"  --> sample: {text[:300]}")

            # ---- checkpoint -------------------------------------------------
            if (step + 1) % cfg.checkpoint_every == 0 or (step + 1) == total_steps:
                p = save_checkpoint(out_dir / "latest.pt", model, optimizer, scaler,
                                    step + 1, rng, {"history": history})
                msg = f"  --> checkpoint at step {step + 1:,}"
                if push_to_hub:
                    try:
                        push_checkpoint(p)
                        msg += f" (pushed to {hub.ckpt_repo})"
                    except Exception as e:
                        msg += f" (Hub push FAILED: {e})"
                print(msg)

    finally:
        logf.close()

    elapsed = time.time() - t_start
    print(f"\n  done in {elapsed / 60:.1f} min")
    (out_dir / "history.json").write_text(json.dumps(history, indent=2, default=str), encoding="utf-8")
    return model, history


def smoke_test(sp, train_tokens, val_tokens, out_dir: Path = Path("checkpoints/smoke"), device: str = "cuda"):
    """A 50-step rehearsal of the real thing.

    Runs the same ``train()`` with a tiny step budget so every code path --
    accumulation, scaler, clipping, eval, sampling, checkpoint save, Hub push,
    resume -- executes before committing 45 minutes of GPU time. Bugs in this
    loop are cheap to find here and expensive to find later.
    """
    print("SMOKE RUN -- rehearsing every code path at 1/100th scale\n")
    model, history = train(
        train_tokens, val_tokens, sp=sp,
        out_dir=Path(out_dir), device=device,
        resume=False, push_to_hub=False,
        max_steps=train_cfg.smoke_max_steps,
    )
    losses = [h["train_loss"] for h in history if h.get("train_loss")]
    if len(losses) >= 2 and losses[-1] >= losses[0]:
        print(f"\n  WARNING: loss did not fall ({losses[0]:.3f} -> {losses[-1]:.3f}). "
              "50 steps is short, but check the LR before starting the real run.")
    else:
        print(f"\n  loss {losses[0]:.3f} -> {losses[-1]:.3f}; all code paths executed.")
    return model, history
