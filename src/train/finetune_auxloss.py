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


# ----------------------------
# sanity helpers
# ----------------------------
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


def _build_loaders(cfg: TrainConfig) -> DataLoader:
    ds = SynthTextDataset(n=cfg.train_samples, image_size=cfg.image_size, seed=cfg.seed)
    dl = DataLoader(
        ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=True,
        drop_last=True,
        collate_fn=collate_fn,
    )
    return dl

import torch
import torch.nn.functional as F

def _encode_cap_feats(pipe: ZImagePipeline, texts: List[str], device: str):
    tok = pipe.tokenizer(
        texts,
        padding=True,
        truncation=True,
        return_tensors="pt",
    )
    tok = {k: v.to(device) for k, v in tok.items()}

    out = pipe.text_encoder(**tok)
    cap = getattr(out, "last_hidden_state", out[0])  # (B, seq, dim)

    # ✅ transformer expects list length B, each item is (seq, dim)
    cap_list = [cap[i] for i in range(cap.shape[0])]
    return cap_list


def _sample_sigma_and_noisy_latents(pipe, latents: torch.Tensor):
    """
    FlowMatchEulerDiscreteScheduler is sigma-based. We'll sample a sigma and add noise.
    Returns: noisy_latents, sigma_tensor
    """
    device = latents.device
    dtype = latents.dtype

    # Ensure scheduler has sigmas; if not, try set_timesteps to populate.
    if not hasattr(pipe.scheduler, "sigmas") or pipe.scheduler.sigmas is None:
        # Fallback: populate timesteps/sigmas
        nT = getattr(getattr(pipe.scheduler, "config", None), "num_train_timesteps", 1000)
        pipe.scheduler.set_timesteps(nT, device=device)

    sigmas = pipe.scheduler.sigmas.to(device=device)

    # sample an index in [0, len(sigmas)-1]
    idx = torch.randint(low=0, high=sigmas.shape[0], size=(latents.shape[0],), device=device)
    sigma = sigmas[idx].to(dtype=dtype)  # (B,)

    noise = torch.randn_like(latents)
    # broadcast sigma -> (B,1,1,1)
    sigma_b = sigma.view(-1, 1, 1, 1)
    noisy = latents + noise * sigma_b

    return noisy, sigma


def forward_generate_decoded_images(pipe: ZImagePipeline, texts: List[str], image_size: int) -> torch.Tensor:
    device = next(pipe.transformer.parameters()).device
    dtype = next(pipe.transformer.parameters()).dtype
    B = len(texts)

    cap_feats = _encode_cap_feats(pipe, texts, device=device)  # list length B, each (seq, dim)

    latent_h = image_size // 8
    latent_w = image_size // 8
    latents = torch.randn((B, 4, latent_h, latent_w), device=device, dtype=dtype)

    noisy_latents, sigma = _sample_sigma_and_noisy_latents(pipe, latents)  # noisy: (B,4,h,w), sigma: (B,)

    # ✅ transformer expects x as list length B, each item is (C,H,W)
    x_list = [noisy_latents[i] for i in range(B)]

    out = pipe.transformer(
        x=x_list,
        t=sigma,          # keep as (B,) tensor
        cap_feats=cap_feats,
        return_dict=True,
    )

    pred = out.sample if hasattr(out, "sample") else out

    # ✅ output may also be list length B of (C,H,W); stack back to (B,C,H,W)
    if isinstance(pred, (list, tuple)):
        pred_latents = torch.stack(pred, dim=0)
    else:
        pred_latents = pred

    sf = getattr(getattr(pipe.vae, "config", None), "scaling_factor", 0.18215)
    decoded = pipe.vae.decode(pred_latents / sf).sample  # [-1,1]
    decoded = (decoded.clamp(-1, 1) + 1) / 2             # [0,1]
    return decoded

def train(cfg: TrainConfig) -> None:
    device = cfg.device if torch.cuda.is_available() else "cpu"
    _seed_all(cfg.seed)

    os.makedirs(cfg.out_dir, exist_ok=True)
    os.makedirs(cfg.ckpt_dir, exist_ok=True)

    pipe = _build_pipe(cfg, device=device)
    model = pipe.transformer
    model.train()

    # NOTE: CLIP loss is frozen model, grads flow to decoded (and thus generator)
    aux_loss_fn = CLIPTextRegionLoss(device=device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr)

    use_amp = (device == "cuda") and (_resolve_dtype(cfg.dtype) in (torch.float16, torch.bfloat16))
    scaler = torch.cuda.amp.GradScaler(enabled=(use_amp and _resolve_dtype(cfg.dtype) == torch.float16))

    # For logging parameter updates
    probe_param = _pick_probe_param(model)

    # Dataloader
    train_loader = _build_loaders(cfg)
    train_iter = iter(train_loader)

    optimizer.zero_grad(set_to_none=True)

    global_step = 0
    micro_step = 0

    while global_step < cfg.num_steps:
        try:
            batch = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            batch = next(train_iter)

        micro_step += 1

        pixel_values: torch.Tensor = batch["pixel_values"].to(device, non_blocking=True)  # (B,3,H,W)
        bboxes: torch.Tensor = batch["bboxes"].to(device, non_blocking=True)              # (B,4)
        texts: List[str] = batch["texts"]

        # snapshot for Δw (only at start of accumulation cycle to keep it meaningful)
        if micro_step % cfg.grad_accum == 1:
            before = probe_param.detach().float().clone()
        else:
            before = None

        # ----------------------------
        # Forward: two modes
        # ----------------------------
        if cfg.sanity_mode:
            # "proof-of-update" loss. Cheap and guarantees weight updates if optimizer works.
            # We only use a few parameter tensors to avoid huge compute.
            with torch.cuda.amp.autocast(enabled=use_amp):
                loss = 0.0
                c = 0
                for p in model.parameters():
                    if not p.requires_grad:
                        continue
                    loss = loss + p.float().pow(2).mean()
                    c += 1
                    if c >= 8:
                        break
                loss = loss / max(c, 1)
                # scale by grad_accum for proper accumulation
                loss = loss / cfg.grad_accum

        else:
            # REAL mode: you must produce decoded images from the model in a differentiable way.
            # If you paste the original forward from your previous training code,
            # I will replace this block with the correct implementation.
            decoded = forward_generate_decoded_images(pipe, texts=texts, image_size=cfg.image_size)
            with torch.cuda.amp.autocast(enabled=use_amp):
                aux_loss = aux_loss_fn(decoded, bboxes=bboxes, texts=texts)
                loss = (cfg.lambda_aux * aux_loss) / cfg.grad_accum

        # ----------------------------
        # Backward (AMP-safe)
        # ----------------------------
        if scaler.is_enabled():
            scaler.scale(loss).backward()
        else:
            loss.backward()

        # ----------------------------
        # Optimizer step every grad_accum micro-steps
        # ----------------------------
        if micro_step % cfg.grad_accum == 0:
            global_step += 1

            # unscale before clipping if using fp16 scaler
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

            # compute Δw once per optimizer step
            after = probe_param.detach().float()
            delta_w = (after - before).abs().mean().item() if before is not None else float("nan")

            if (global_step % cfg.log_every) == 0 or global_step <= 3:
                print(
                    f"[step {global_step}/{cfg.num_steps}] "
                    f"loss={float(loss.detach().cpu().item() * cfg.grad_accum):.6f} "
                    f"grad_norm={grad_norm_val:.4e} mean|Δw|={delta_w:.4e} "
                    f"(sanity_mode={cfg.sanity_mode})"
                )

            # optional memory cleanup
            if device == "cuda" and (global_step % (cfg.log_every * 4) == 0):
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
