from __future__ import annotations

from pathlib import Path
from typing import Optional

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.config import TrainConfig
from src.utils.seed import seed_everything
from src.utils.io import ensure_dir
from src.data.synth_text_dataset import SynthTextDataset, collate_fn
from src.losses.clip_text_region_loss import CLIPTextRegionLoss
from src.zimage.pipeline_utils import load_zimage


def _get_components(pipe):
    transformer = getattr(pipe, "transformer", None)
    vae = getattr(pipe, "vae", None)
    scheduler = getattr(pipe, "scheduler", None)
    tokenizer = getattr(pipe, "tokenizer", None)
    text_encoder = getattr(pipe, "text_encoder", None)

    missing = [
        name
        for name, obj in [
            ("transformer", transformer),
            ("vae", vae),
            ("scheduler", scheduler),
            ("tokenizer", tokenizer),
            ("text_encoder", text_encoder),
        ]
        if obj is None
    ]

    if missing:
        raise RuntimeError(
            "Missing expected pipeline components: "
            + ", ".join(missing)
            + ". ZImagePipeline API might have changed; inspect pipe components."
        )

    return transformer, vae, scheduler, tokenizer, text_encoder


def _call_zimage_transformer(
    transformer,
    noisy_latents: torch.Tensor,  # (B,C,H,W)
    t: torch.Tensor,              # (B,)
    enc: torch.Tensor,            # (B,S,D)
) -> torch.Tensor:
    """
    ZImageTransformer2DModel.forward signature observed:
      (x: List[Tensor] or List[List[Tensor]]],
       t,
       cap_feats: List[Tensor] or List[List[Tensor]]],
       ...)

    Transformer expects:
      - each x[i] is (C, F, H, W) for one sample -> we use F=1
      - each cap_feats[i] is (S, D) for one sample

    Returns:
      pred latents (B,C,H,W)  (if the model returns (B,C,F,H,W) we squeeze F=1)
    """
    if noisy_latents.dim() != 4:
        raise RuntimeError(f"Expected noisy_latents (B,C,H,W), got {tuple(noisy_latents.shape)}")
    if enc.dim() != 3:
        raise RuntimeError(f"Expected enc (B,S,D), got {tuple(enc.shape)}")
    B = noisy_latents.shape[0]
    if enc.shape[0] != B:
        raise RuntimeError(f"Batch mismatch: latents B={B} vs enc B={enc.shape[0]}")

    # Per-sample list with frames dim F=1: (C,1,H,W)
    x = [noisy_latents[i].unsqueeze(1) for i in range(B)]
    cap_feats = [enc[i] for i in range(B)]

    out = transformer(x=x, t=t, cap_feats=cap_feats, return_dict=True)

    # Extract ONLY latent prediction (strict)
    v: Optional[object] = None
    if hasattr(out, "sample"):
        v = out.sample
    elif hasattr(out, "x"):
        v = out.x
    elif hasattr(out, "pred"):
        v = out.pred
    elif isinstance(out, dict):
        for k in ("sample", "x", "pred", "eps"):
            if k in out:
                v = out[k]
                break

    if v is None:
        raise RuntimeError(f"Could not extract latent prediction from transformer output type={type(out)}")

    # If model returns list per sample -> stack
    if isinstance(v, (list, tuple)):
        if len(v) != B:
            raise RuntimeError(f"Expected list output length B={B}, got len={len(v)}")
        v0 = v[0]
        if not torch.is_tensor(v0):
            raise RuntimeError(f"Expected tensor elements in output list, got {type(v0)}")

        if v0.dim() == 4:          # (C,F,H,W) per sample
            v = torch.stack(list(v), dim=0)  # (B,C,F,H,W)
        elif v0.dim() == 3:        # (C,H,W) per sample
            v = torch.stack(list(v), dim=0)  # (B,C,H,W)
        else:
            raise RuntimeError(f"Unexpected output element shape: {tuple(v0.shape)}")

    if not torch.is_tensor(v):
        raise RuntimeError(f"Expected tensor prediction, got {type(v)}")

    # Squeeze frames if present: (B,C,1,H,W) -> (B,C,H,W)
    if v.dim() == 5 and v.shape[2] == 1:
        v = v.squeeze(2)

    if v.dim() != 4:
        raise RuntimeError(f"Transformer prediction has unexpected shape: {tuple(v.shape)}")

    return v


def _enable_gradient_checkpointing(transformer) -> None:
    # Different modules expose it differently
    if hasattr(transformer, "enable_gradient_checkpointing"):
        try:
            transformer.enable_gradient_checkpointing()
            return
        except Exception:
            pass
    if hasattr(transformer, "gradient_checkpointing"):
        try:
            transformer.gradient_checkpointing = True
        except Exception:
            pass


def train(cfg: TrainConfig) -> None:
    seed_everything(cfg.seed)
    ensure_dir(cfg.out_dir)
    ensure_dir(cfg.ckpt_dir)

    device = torch.device(cfg.device)

    pipe = load_zimage(cfg.model_id, dtype=cfg.dtype, device=cfg.device)
    transformer, vae, scheduler, tokenizer, text_encoder = _get_components(pipe)

    # Freeze heavy parts
    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)
    transformer.train()

    # Try to reduce VRAM
    _enable_gradient_checkpointing(transformer)

    # dtypes
    vae_dtype = next(vae.parameters()).dtype  # bf16/fp16 based on pipeline load

    # autocast dtype for cuda
    if str(vae_dtype) == "torch.bfloat16":
        ac_dtype = torch.bfloat16
    elif str(vae_dtype) == "torch.float16":
        ac_dtype = torch.float16
    else:
        ac_dtype = None  # autocast disabled

    autocast_enabled = ac_dtype is not None and device.type == "cuda"

    # -------------------------
    # Optimizer (memory-aware)
    # -------------------------
    params = [p for p in transformer.parameters() if p.requires_grad]

    try:
        import bitsandbytes as bnb
        opt = bnb.optim.AdamW8bit(params, lr=cfg.lr)
        print("✅ Using AdamW8bit (bitsandbytes)")
    except Exception as e:
        opt = torch.optim.SGD(params, lr=cfg.lr, momentum=0.9)
        print("⚠️ bitsandbytes not available -> using SGD")
        print("   reason:", repr(e))
    # Data
    ds = SynthTextDataset(n=cfg.train_samples, image_size=cfg.image_size, seed=cfg.seed)
    dl = DataLoader(
        ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
    )
    dl_iter = iter(dl)

    # Auxiliary loss (CLIP-based)
    aux_loss_fn = CLIPTextRegionLoss(device=cfg.device)

    # How often to compute aux loss (memory saver)
    aux_every = getattr(cfg, "aux_every", 4)  # if not in config, default=4

    step = 0
    pbar = tqdm(total=cfg.num_steps, desc="train")

    while step < cfg.num_steps:
        # -------------------------
        # Fetch batch
        # -------------------------
        try:
            batch = next(dl_iter)
        except StopIteration:
            dl_iter = iter(dl)
            batch = next(dl_iter)

        pixel_values = batch["pixel_values"].to(device, non_blocking=True)  # float32
        texts = batch["texts"]
        bboxes = batch["bboxes"].to(device, non_blocking=True)

        # match dtype for VAE
        pixel_values = pixel_values.to(dtype=vae_dtype)

        # -------------------------
        # Encode to latents (no grad)
        # -------------------------
        with torch.no_grad():
            with torch.autocast(device_type="cuda", dtype=ac_dtype, enabled=autocast_enabled):
                latents = vae.encode(pixel_values * 2.0 - 1.0).latent_dist.sample()
                latents = latents * vae.config.scaling_factor

        latents = latents.to(dtype=vae_dtype)

        # -------------------------
        # Flow-like noising (scheduler-agnostic)
        # -------------------------
        noise = torch.randn_like(latents)  # (B,C,H,W)
        bsz = latents.shape[0]

        # continuous time in [0,1]
        t = torch.rand((bsz,), device=device, dtype=vae_dtype)  # (B,)

        # x_t = x + t * eps
        noisy_latents = latents + t.view(-1, 1, 1, 1) * noise
        noisy_latents = noisy_latents.to(dtype=vae_dtype)

        # -------------------------
        # Text conditioning (no grad)
        # -------------------------
        with torch.no_grad():
            text_inputs = tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
            text_inputs = {k: v.to(device) for k, v in text_inputs.items()}
            enc = text_encoder(**text_inputs).last_hidden_state

        enc = enc.to(dtype=noisy_latents.dtype)

        # -------------------------
        # Transformer forward (grad)
        # -------------------------
        with torch.autocast(device_type="cuda", dtype=ac_dtype, enabled=autocast_enabled):
            pred = _call_zimage_transformer(transformer, noisy_latents, t.to(dtype=noisy_latents.dtype), enc)
        pred = pred.to(dtype=noisy_latents.dtype)

        # -------------------------
        # Diffusion proxy loss
        # -------------------------
        if pred.shape != noise.shape:
            raise RuntimeError(f"pred shape {tuple(pred.shape)} != noise shape {tuple(noise.shape)}")

        diffusion_loss = F.mse_loss(pred.float(), noise.float())

        # -------------------------
        # Auxiliary loss (expensive) every N steps
        # -------------------------
        do_aux = (cfg.lambda_aux > 0) and (step % int(aux_every) == 0)

        if do_aux:
            # proxy "denoised" latents (keeps gradients connected)
            latents_hat = (noisy_latents - pred).to(dtype=vae_dtype)
            latents_hat = latents_hat / vae.config.scaling_factor

            # VAE decode (grad flows from aux -> decoded -> latents_hat -> pred -> transformer)
            with torch.autocast(device_type="cuda", dtype=ac_dtype, enabled=autocast_enabled):
                decoded = vae.decode(latents_hat).sample  # [-1,1]

            decoded = (decoded + 1.0) / 2.0
            decoded = decoded.clamp(0, 1)

            # CLIP loss
            aux_loss = aux_loss_fn(decoded, bboxes=bboxes, texts=texts)
        else:
            aux_loss = torch.zeros((), device=device, dtype=torch.float32)

        # -------------------------
        # Combined loss + backward
        # -------------------------
        loss = diffusion_loss + cfg.lambda_aux * aux_loss
        loss = loss / cfg.grad_accum
        loss.backward()

        if (step + 1) % cfg.grad_accum == 0:
            torch.nn.utils.clip_grad_norm_(transformer.parameters(), cfg.max_grad_norm)
            opt.step()
            opt.zero_grad(set_to_none=True)

        # -------------------------
        # Logging
        # -------------------------
        if step % 25 == 0:
            pbar.set_postfix(
                {
                    "diff": float(diffusion_loss.detach().cpu()),
                    "aux": float(aux_loss.detach().cpu()) if do_aux else 0.0,
                    "lam": float(cfg.lambda_aux),
                    "aux_every": int(aux_every),
                    "dtype": str(vae_dtype).replace("torch.", ""),
                }
            )

        # -------------------------
        # Checkpoint
        # -------------------------
        if step > 0 and step % 200 == 0:
            ckpt_path = Path(cfg.ckpt_dir) / f"transformer_step_{step}.pt"
            torch.save(transformer.state_dict(), ckpt_path)

        # free big tensors explicitly (helps fragmentation)
        if do_aux:
            del decoded, latents_hat
        del latents, noisy_latents, noise, pred, enc

        step += 1
        pbar.update(1)

    pbar.close()

    ckpt_path = Path(cfg.ckpt_dir) / "transformer_final.pt"
    torch.save(transformer.state_dict(), ckpt_path)
    print(f"Saved: {ckpt_path}")


if __name__ == "__main__":
    cfg = TrainConfig()
    train(cfg)
