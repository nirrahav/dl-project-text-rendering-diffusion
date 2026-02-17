from __future__ import annotations

import os
import gc
import math
import random
from typing import Any, Dict, List, Tuple

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from diffusers import ZImagePipeline

from src.losses.clip_text_region_loss import CLIPTextRegionLoss
from src.config import TrainConfig

# UPDATE THIS IMPORT PATH to where you saved SynthTextDataset
from src.data.synth_text_dataset import SynthTextDataset, collate_fn

SANITY_MODE = False   # True = verify optimizer updates weights, False = real CLIP aux training
LOG_EVERY = 25        # print every N optimizer steps
DEBUG_SHAPES_ONCE = False  # set True for one run if still failing


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
# Z-Image differentiable forward (matches your transformer signature)
#   forward(x: list[Tensor,...], t, cap_feats: list[Tensor,...], ...)
# ============================================================

def _encode_cap_feats(pipe: ZImagePipeline, texts: List[str], device: str) -> List[torch.Tensor]:
    """
    Must return: list length B, each tensor is 2D (seq_len, hidden_dim).
    This is REQUIRED for transformer_z_image._pad_with_ids which does repeat(pad_len, 1).
    """
    tok = pipe.tokenizer(
        texts,
        padding=True,
        truncation=True,
        return_tensors="pt",
    )
    tok = {k: v.to(device) for k, v in tok.items()}

    out = pipe.text_encoder(**tok)
    cap = getattr(out, "last_hidden_state", out[0])  # expected (B, seq, dim)

    # Ensure we have (B, seq, dim)
    if cap.dim() == 4 and cap.shape[1] == 1:
        cap = cap.squeeze(1)  # (B, seq, dim)
    if cap.dim() != 3:
        raise RuntimeError(f"Unexpected cap shape from text encoder: {tuple(cap.shape)}")

    cap_list: List[torch.Tensor] = []
    B = cap.shape[0]
    for i in range(B):
        c = cap[i]  # should be (seq, dim)

        # Clean up any weird singleton dims
        while c.dim() > 2 and c.shape[0] == 1:
            c = c.squeeze(0)
        if c.dim() == 3 and c.shape[1] == 1:
            c = c.squeeze(1)
        if c.dim() > 2:
            # last resort: flatten all but seq
            c = c.reshape(c.shape[0], -1)

        if c.dim() != 2:
            raise RuntimeError(f"cap_feat must be 2D (seq, dim). got shape={tuple(c.shape)}")

        cap_list.append(c)

    return cap_list


def _ensure_scheduler_sigmas(pipe: ZImagePipeline, device: str) -> None:
    """
    FlowMatchEulerDiscreteScheduler is sigma-based. Make sure .sigmas exists.
    """
    if not hasattr(pipe.scheduler, "sigmas") or pipe.scheduler.sigmas is None:
        nT = getattr(getattr(pipe.scheduler, "config", None), "num_train_timesteps", 1000)
        pipe.scheduler.set_timesteps(nT, device=device)


def _sample_sigma_and_noisy_latents(pipe: ZImagePipeline, latents: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Sample a sigma from scheduler.sigmas and add noise:
      noisy = latents + sigma * eps
    Returns:
      noisy_latents: same shape as latents
      sigma: (B,) sigma values
    """
    device = latents.device
    dtype = latents.dtype

    _ensure_scheduler_sigmas(pipe, device=device)
    sigmas = pipe.scheduler.sigmas.to(device=device)

    idx = torch.randint(low=0, high=sigmas.shape[0], size=(latents.shape[0],), device=device)
    sigma = sigmas[idx].to(dtype=dtype)  # (B,)

    eps = torch.randn_like(latents)
    noisy = latents + eps * sigma.view(-1, 1, 1, 1)
    return noisy, sigma


def forward_generate_decoded_images(pipe: ZImagePipeline, texts: List[str], image_size: int) -> torch.Tensor:
    """
    Differentiable forward for Z-Image-Turbo (your version):
      - transformer expects x as list of (C,F,H,W) where C=16, F=1
      - patch_size fixed at 2, f_patch_size fixed at 1 (per all_patch_size/all_f_patch_size)
      - vae expects latent_channels=16, so we can decode directly.

    Returns:
      decoded images in [0,1], shape (B,3,image_size,image_size)
    """
    device = next(pipe.transformer.parameters()).device
    dtype = next(pipe.transformer.parameters()).dtype
    B = len(texts)

    # ---- encode text -> cap_feats list length B, each (seq, dim) ----
    cap_feats = _encode_cap_feats(pipe, texts, device=device)  # IMPORTANT: each item must be 2D

    # ---- sample latents with correct channel count (C=16) ----
    in_ch = int(getattr(pipe.transformer.config, "in_channels", 16))  # confirmed 16
    latent_h = image_size // getattr(pipe, "vae_scale_factor", 8)     # confirmed 8 -> 64 when image_size=512
    latent_w = image_size // getattr(pipe, "vae_scale_factor", 8)

    latents = torch.randn((B, in_ch, latent_h, latent_w), device=device, dtype=dtype)

    # ---- add sigma noise ----
    noisy_latents, sigma = _sample_sigma_and_noisy_latents(pipe, latents)  # (B,16,h,w), (B,)

    # ---- transformer expects (C,F,H,W). For static image, F=1 ----
    noisy_latents = noisy_latents.unsqueeze(2)  # (B,16,1,h,w)
    x_list = [noisy_latents[i] for i in range(B)]  # list length B, each (16,1,h,w)

    # ---- forward transformer ----
    out = pipe.transformer(
        x=x_list,
        t=sigma,               # (B,)
        cap_feats=cap_feats,   # list length B
        return_dict=True,
        patch_size=2,          # only supported value
        f_patch_size=1,        # only supported value
    )

    pred = out.sample if hasattr(out, "sample") else out

    # pred usually list length B of (16,1,h,w). Stack -> (B,16,1,h,w)
    if isinstance(pred, (list, tuple)):
        pred_latents = torch.stack(pred, dim=0)
    else:
        pred_latents = pred

    # Drop F dimension -> (B,16,h,w)
    if pred_latents.dim() == 5:
        pred_latents = pred_latents[:, :, 0]

    # ---- VAE decode ----
    sf = getattr(getattr(pipe.vae, "config", None), "scaling_factor", 1.0)
    decoded_raw = pipe.vae.decode(pred_latents / sf).sample  # unbounded / roughly [-?, ?]

    # ✅ Differentiable squashing to [0,1] (keeps gradients alive)
    decoded = torch.sigmoid(decoded_raw)

    return decoded



# ============================================================
# Train
# ============================================================

def train(cfg: TrainConfig) -> None:
    device = cfg.device if torch.cuda.is_available() else "cpu"
    _seed_all(cfg.seed)

    os.makedirs(cfg.out_dir, exist_ok=True)
    os.makedirs(cfg.ckpt_dir, exist_ok=True)

    pipe = _build_pipe(cfg, device=device)
    model = pipe.transformer
    model.train()

    aux_loss_fn = CLIPTextRegionLoss(device=device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr)

    dtype = _resolve_dtype(cfg.dtype)
    use_autocast = (device == "cuda") and (dtype in (torch.float16, torch.bfloat16))
    scaler = torch.cuda.amp.GradScaler(enabled=(device == "cuda" and dtype == torch.float16))

    probe_param = _pick_probe_param(model)

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

        # ----------------------------
        # Forward
        # ----------------------------
        if SANITY_MODE:
            with torch.cuda.amp.autocast(enabled=use_autocast):
                loss = 0.0
                c = 0
                for p in model.parameters():
                    if not p.requires_grad:
                        continue
                    loss = loss + p.float().pow(2).mean()
                    c += 1
                    if c >= 8:
                        break
                loss = (loss / max(c, 1)) / cfg.grad_accum
        else:
            # compute aux every aux_every micro-steps
            if (micro_step % cfg.aux_every) == 0:
                decoded = forward_generate_decoded_images(pipe, texts=texts, image_size=cfg.image_size)
                with torch.cuda.amp.autocast(enabled=use_autocast):
                    aux = aux_loss_fn(decoded, bboxes=bboxes, texts=texts)
                    loss = (cfg.lambda_aux * aux) / cfg.grad_accum
            else:
                loss = (probe_param.sum() * 0.0) / cfg.grad_accum  # dummy loss to check optimizer updates weights when aux is not computed
        # ----------------------------
        # Backward
        # ----------------------------
        if scaler.is_enabled():
            scaler.scale(loss).backward()
        else:
            loss.backward()

        # ----------------------------
        # Optimizer step (every grad_accum micro-steps)
        # ----------------------------
        if micro_step % cfg.grad_accum == 0:
            global_step += 1

            # measure Δw on the actual optimizer step (never NaN)
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

            if global_step <= 3 or (global_step % LOG_EVERY == 0):
                print(
                    f"[step {global_step}/{cfg.num_steps}] "
                    f"loss={float(loss.detach().cpu().item() * cfg.grad_accum):.6f} "
                    f"grad_norm={grad_norm_val:.4e} mean|Δw|={delta_w:.4e} "
                    f"(SANITY_MODE={SANITY_MODE})"
                )

            if device == "cuda" and (global_step % (LOG_EVERY * 4) == 0):
                torch.cuda.empty_cache()
                gc.collect()

    # ----------------------------
    # Save checkpoint
    # ----------------------------
    ckpt_path = os.path.join(cfg.ckpt_dir, "transformer_final.pt")
    torch.save(model.state_dict(), ckpt_path)
    print("Saved:", ckpt_path)

    # ----------------------------
    # Verify differs from baseline
    # ----------------------------
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