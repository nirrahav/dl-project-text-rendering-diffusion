from __future__ import annotations

import os
import gc
import random
from typing import Dict, List

import torch
from torch.utils.data import DataLoader
from diffusers import ZImagePipeline

from src.losses.clip_text_region_loss import CLIPTextRegionLoss
from src.data.synth_text_dataset import SynthTextDataset, collate_fn
from src.config import TrainConfig  # <-- change this import if your TrainConfig lives elsewhere


# ----------------------------
# dtype helper
# ----------------------------
def _resolve_dtype(dtype_str: str) -> torch.dtype:
    d = dtype_str.lower().strip()
    if d in ("fp16", "float16"):
        return torch.float16
    if d in ("bf16", "bfloat16"):
        return torch.bfloat16
    if d in ("fp32", "float32"):
        return torch.float32
    raise ValueError(f"Unknown dtype: {dtype_str}")


def _seed_all(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _pick_probe_param(module: torch.nn.Module) -> torch.nn.Parameter:
    for p in module.parameters():
        if p.requires_grad:
            return p
    raise RuntimeError("No trainable parameters found.")


@torch.no_grad()
def _mean_abs_param_diff(state_a: Dict[str, torch.Tensor], state_b: Dict[str, torch.Tensor], n_keys: int = 200) -> float:
    keys = list(state_a.keys())[: min(n_keys, len(state_a))]
    diffs = []
    for k in keys:
        diffs.append((state_a[k].float() - state_b[k].float()).abs().mean().item())
    return float(sum(diffs) / max(len(diffs), 1))


def _build_pipe(cfg: TrainConfig, device: str) -> ZImagePipeline:
    torch_dtype = _resolve_dtype(cfg.dtype)
    pipe = ZImagePipeline.from_pretrained(
        cfg.model_id,
        torch_dtype=torch_dtype,
        low_cpu_mem_usage=True,
    ).to(device)
    pipe.set_progress_bar_config(disable=True)
    return pipe


def _build_loader(cfg: TrainConfig) -> DataLoader:
    ds = SynthTextDataset(n=cfg.train_samples, image_size=cfg.image_size, seed=cfg.seed)
    return DataLoader(
        ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=True,
        drop_last=True,
        collate_fn=collate_fn,
    )


# ============================================================
# Z-Image differentiable forward (ASSUMES you already pasted the working
# forward_generate_decoded_images + _encode_cap_feats + sigma sampling)
# ============================================================
# IMPORTANT: this function must exist in this file (or be imported) already.
# from src.train.finetune_auxloss import forward_generate_decoded_images


# ============================================================
# Train (FIXED AMP: torch.amp.autocast + torch.amp.GradScaler)
# ============================================================
def train(cfg: TrainConfig) -> None:
    device = cfg.device if torch.cuda.is_available() else "cpu"
    _seed_all(cfg.seed)

    os.makedirs(cfg.out_dir, exist_ok=True)
    os.makedirs(cfg.ckpt_dir, exist_ok=True)

    # ---- build ----
    pipe = _build_pipe(cfg, device=device)
    model = pipe.transformer
    model.train()

    aux_loss_fn = CLIPTextRegionLoss(device=device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr)

    # ---- AMP setup (PyTorch 2.6+ safe) ----
    dtype = _resolve_dtype(cfg.dtype)
    use_amp = (device == "cuda") and (dtype in (torch.float16, torch.bfloat16))
    use_scaler = (device == "cuda") and (dtype == torch.float16)

    from torch.amp import autocast, GradScaler
    scaler = GradScaler(enabled=use_scaler)

    # ---- logging param ----
    probe_param = _pick_probe_param(model)

    # ---- data ----
    loader = _build_loader(cfg)
    it = iter(loader)

    optimizer.zero_grad(set_to_none=True)

    global_step = 0
    micro_step = 0

    while global_step < cfg.num_steps:
        try:
            batch = next(it)
        except StopIteration:
            it = iter(loader)
            batch = next(it)

        micro_step += 1

        bboxes: torch.Tensor = batch["bboxes"].to(device, non_blocking=True)
        texts: List[str] = batch["texts"]

        # ---- forward ----
        # compute aux loss every aux_every micro-steps; otherwise do 0*param.sum() so backward won't crash
        if (micro_step % cfg.aux_every) == 0:
            decoded = forward_generate_decoded_images(pipe, texts=texts, image_size=cfg.image_size)

            with autocast(device_type="cuda", enabled=use_amp, dtype=dtype):
                aux = aux_loss_fn(decoded, bboxes=bboxes, texts=texts)
                loss = (cfg.lambda_aux * aux) / cfg.grad_accum
        else:
            # ✅ "zero" but connected to graph
            loss = (probe_param.sum() * 0.0) / cfg.grad_accum

        # ---- backward ----
        if scaler.is_enabled():
            scaler.scale(loss).backward()
        else:
            loss.backward()

        # ---- optimizer step ----
        if micro_step % cfg.grad_accum == 0:
            global_step += 1

            before = probe_param.detach().float().clone()

            if scaler.is_enabled():
                scaler.unscale_(optimizer)

            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.max_grad_norm)
            grad_norm_val = float(grad_norm.detach().cpu().item()) if torch.is_tensor(grad_norm) else float(grad_norm)

            if scaler.is_enabled():
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()

            optimizer.zero_grad(set_to_none=True)

            after = probe_param.detach().float()
            delta_w = (after - before).abs().mean().item()

            # If your TrainConfig doesn't have log_every, just hardcode:
            log_every = getattr(cfg, "log_every", 20)
            if global_step <= 3 or (global_step % log_every == 0):
                print(
                    f"[step {global_step}/{cfg.num_steps}] "
                    f"loss={float(loss.detach().cpu().item() * cfg.grad_accum):.6f} "
                    f"grad_norm={grad_norm_val:.4e} mean|Δw|={delta_w:.4e}"
                )

            if device == "cuda" and (global_step % (log_every * 4) == 0):
                torch.cuda.empty_cache()
                gc.collect()

    # ---- save ----
    ckpt_path = os.path.join(cfg.ckpt_dir, "transformer_final.pt")
    torch.save(model.state_dict(), ckpt_path)
    print("Saved:", ckpt_path)

    # ---- verify differs from baseline ----
    trained_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}

    baseline = ZImagePipeline.from_pretrained(
        cfg.model_id,
        torch_dtype=_resolve_dtype(cfg.dtype),
    ).to("cpu")
    baseline_state = {k: v.detach().cpu() for k, v in baseline.transformer.state_dict().items()}

    diff = _mean_abs_param_diff(trained_state, baseline_state)
    print("Diff(trained vs baseline) mean(|Δ|) ~", diff)

    if diff == 0.0:
        raise RuntimeError("Final checkpoint is identical to baseline (diff=0). Training likely did not update weights.")