from dataclasses import dataclass

@dataclass
class TrainConfig:
    # Model
    model_id: str = "Tongyi-MAI/Z-Image-Turbo"
    dtype: str = "bf16"        # "fp16" / "bf16" / "fp32"
    device: str = "cuda"

    # Data
    image_size: int = 512
    train_samples: int = 2000
    val_samples: int = 128
    batch_size: int = 2
    num_workers: int = 2

    # Training
    lr: float = 1e-5
    num_steps: int = 800
    grad_accum: int = 4
    max_grad_norm: float = 1.0
    aux_every: int = 4

    # Loss weights
    lambda_aux: float = 0.2

    # Output
    out_dir: str = "outputs"
    ckpt_dir: str = "checkpoints"
    seed: int = 42

    # Debug / sanity
    sanity_mode: bool = True   # True = verify optimizer updates weights (no diffusion forward needed yet)
    log_every: int = 25
