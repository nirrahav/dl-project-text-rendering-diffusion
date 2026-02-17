from dataclasses import dataclass

@dataclass
class TrainConfig:
    # =========================================================
    # Model
    # =========================================================
    model_id: str = "Tongyi-MAI/Z-Image-Turbo"
    dtype: str = "bf16"           # "fp16" / "bf16" / "fp32"
    device: str = "cuda"

    # =========================================================
    # Data
    # =========================================================
    image_size: int = 512
    train_samples: int = 2000
    val_samples: int = 128
    batch_size: int = 2
    num_workers: int = 2

    # =========================================================
    # Training
    # =========================================================
    lr: float = 1e-5
    num_steps: int = 800
    grad_accum: int = 4
    max_grad_norm: float = 1.0
    aux_every: int = 4             # apply CLIP aux every N micro steps
    log_every: int = 25            # print every N optimizer steps
    sanity_mode: bool = False      # True = debug updates only, False = real CLIP training

    # =========================================================
    # Loss
    # =========================================================
    lambda_aux: float = 0.2        # weight for text-region auxiliary loss

    # =========================================================
    # Scheduler (optional override)
    # =========================================================
    num_train_timesteps: int = 1000  # used if scheduler needs initialization

    # =========================================================
    # Output
    # =========================================================
    out_dir: str = "outputs"
    ckpt_dir: str = "checkpoints"
    save_name: str = "transformer_final.pt"

    # =========================================================
    # Misc
    # =========================================================
    seed: int = 42
    mixed_precision: bool = True   # enable autocast for bf16/fp16
