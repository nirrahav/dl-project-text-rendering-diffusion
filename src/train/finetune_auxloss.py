from __future__ import annotations

from pathlib import Path

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


def _call_zimage_transformer(transformer, noisy_latents, t, enc):
    """
    ZImageTransformer2DModel.forward expects:
      - x: List[Tensor] where each Tensor is an image/latent for a single sample
      - t: time / timestep (typically shape (B,))
      - cap_feats: List[Tensor] where each Tensor is (seq_len, dim) for a single sample
    """
    # noisy_latents: (B, C, H, W)
    # enc:          (B, S, D)
    B = noisy_latents.shape[0]
    assert enc.shape[0] == B, "Batch size mismatch between latents and caption features"

    # split batch into per-sample items
    x = [noisy_latents[i] for i in range(B)]          # each: (C, H, W)
    cap_feats = [enc[i] for i in range(B)]            # each: (S, D)

    out = transformer(x=x, t=t, cap_feats=cap_feats, return_dict=True)

    # Robustly extract prediction
    if isinstance(out, dict):
        for k in ("sample", "pred", "out", "x", "eps"):
            if k in out:
                v = out[k]
                break
        else:
            v = next(iter(out.values()))
    else:
        if hasattr(out, "sample"):
            v = out.sample
        elif hasattr(out, "pred"):
            v = out.pred
        elif hasattr(out, "x"):
            v = out.x
        else:
            v = out

    # Z-Image may return a list (per-sample) — stack back to (B, C, H, W)
    if isinstance(v, (list, tuple)):
        # if each item is (C,H,W), stack
        if torch.is_tensor(v[0]) and v[0].dim() == 3:
            v = torch.stack(list(v), dim=0)
        # if each item is (1,C,H,W), cat
        elif torch.is_tensor(v[0]) and v[0].dim() == 4:
            v = torch.cat(list(v), dim=0)

    return v


def train(cfg: TrainConfig) -> None:
    seed_everything(cfg.seed)
    ensure_dir(cfg.out_dir)
    ensure_dir(cfg.ckpt_dir)

    # --- Load pipeline ---
    pipe = load_zimage(cfg.model_id, dtype=cfg.dtype, device=cfg.device)
    transformer, vae, scheduler, tokenizer, text_encoder = _get_components(pipe)

    # Freeze VAE + text encoder
    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)
    transformer.train()

    device = torch.device(cfg.device)
    vae_dtype = next(vae.parameters()).dtype  # fp16 / bf16

    # Optimizer
    opt = torch.optim.AdamW(transformer.parameters(), lr=cfg.lr)

    # Dataset
    ds = SynthTextDataset(
        n=cfg.train_samples,
        image_size=cfg.image_size,
        seed=cfg.seed,
    )
    dl = DataLoader(
        ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
    )
    dl_iter = iter(dl)

    # Auxiliary loss
    aux_loss_fn = CLIPTextRegionLoss(device=cfg.device)

    # Training loop
    step = 0
    pbar = tqdm(total=cfg.num_steps, desc="train")

    while step < cfg.num_steps:
        # ------------------------------------------------------------
        # 1) Fetch batch (ALWAYS defined every iteration)
        # ------------------------------------------------------------
        try:
            batch = next(dl_iter)
        except StopIteration:
            dl_iter = iter(dl)
            batch = next(dl_iter)

        pixel_values = batch["pixel_values"].to(device, non_blocking=True)
        texts = batch["texts"]
        bboxes = batch["bboxes"].to(device, non_blocking=True)

        # Match dtype with VAE (critical for fp16/bf16)
        pixel_values = pixel_values.to(dtype=vae_dtype)

        # ------------------------------------------------------------
        # 2) Encode images → latents
        # ------------------------------------------------------------
        with torch.no_grad():
            latents = vae.encode(pixel_values * 2.0 - 1.0).latent_dist.sample()
            latents = latents * vae.config.scaling_factor

        latents = latents.to(dtype=vae_dtype)

        # ------------------------------------------------------------
        # 3) Flow-matching noise + timestep
        # ------------------------------------------------------------
        # Flow-matching style noising (scheduler-agnostic)
        noise = torch.randn_like(latents)

        bsz = latents.shape[0]

        # sample continuous time / noise level in [0, 1]
        t = torch.rand((bsz,), device=device, dtype=vae_dtype)  # (B,)

        # create noisy latents: x_t = x + t * ε
        noisy_latents = latents + t.view(-1, 1, 1, 1) * noise
        noisy_latents = noisy_latents.to(dtype=vae_dtype)

        # feed "t" as timestep conditioning (many flow-match models accept continuous time)
        timesteps = t  # keep name for downstream transformer call

        # ------------------------------------------------------------
        # 4) Text conditioning
        # ------------------------------------------------------------
        with torch.no_grad():
            text_inputs = tokenizer(
                texts,
                padding=True,
                truncation=True,
                return_tensors="pt",
            )
            text_inputs = {k: v.to(device) for k, v in text_inputs.items()}
            enc = text_encoder(**text_inputs).last_hidden_state

        # ------------------------------------------------------------
        # 5) Predict noise / flow
        # ------------------------------------------------------------
        timesteps = timesteps.to(device)

        # timesteps for flow-matching: keep as float tensor (B,)
        # make sure it's on device + dtype consistent
        t = timesteps.to(noisy_latents.device)
        # אם timesteps אצלך כבר float/bf16 זה מצוין. אם הוא long — תעשה t = timesteps.float()
        if t.dtype in (torch.int32, torch.int64):
            t = t.float()
        t = t.to(dtype=noisy_latents.dtype)

        pred = _call_zimage_transformer(transformer, noisy_latents, t, enc)
        pred = pred.to(dtype=noisy_latents.dtype)

        pred = pred.to(dtype=noisy_latents.dtype)

        diffusion_loss = F.mse_loss(pred.float(), noise.float())

        # ------------------------------------------------------------
        # 6) Auxiliary loss (FlowMatch-friendly proxy)
        # ------------------------------------------------------------
        # No alphas_cumprod in FlowMatch schedulers.
        # Use a denoised-latent proxy to keep gradients connected.
        latents_hat = (noisy_latents - pred) / vae.config.scaling_factor
        latents_hat = latents_hat.to(dtype=vae_dtype)

        decoded = vae.decode(latents_hat).sample  # [-1, 1]
        decoded = (decoded + 1.0) / 2.0
        decoded = decoded.clamp(0, 1)

        aux_loss = aux_loss_fn(decoded, bboxes=bboxes, texts=texts)

        # ------------------------------------------------------------
        # 7) Combined loss + backward
        # ------------------------------------------------------------
        loss = diffusion_loss + cfg.lambda_aux * aux_loss
        loss = loss / cfg.grad_accum
        loss.backward()

        if (step + 1) % cfg.grad_accum == 0:
            torch.nn.utils.clip_grad_norm_(
                transformer.parameters(),
                cfg.max_grad_norm,
            )
            opt.step()
            opt.zero_grad(set_to_none=True)

        # Logging
        if step % 25 == 0:
            pbar.set_postfix(
                {
                    "diff": float(diffusion_loss.detach().cpu()),
                    "aux": float(aux_loss.detach().cpu()),
                    "lam": cfg.lambda_aux,
                    "dtype": str(vae_dtype).replace("torch.", ""),
                }
            )

        # Checkpoint
        if step > 0 and step % 200 == 0:
            ckpt_path = Path(cfg.ckpt_dir) / f"transformer_step_{step}.pt"
            torch.save(transformer.state_dict(), ckpt_path)

        step += 1
        pbar.update(1)

    pbar.close()

    ckpt_path = Path(cfg.ckpt_dir) / "transformer_final.pt"
    torch.save(transformer.state_dict(), ckpt_path)
    print(f"Saved: {ckpt_path}")


if __name__ == "__main__":
    cfg = TrainConfig()
    train(cfg)
