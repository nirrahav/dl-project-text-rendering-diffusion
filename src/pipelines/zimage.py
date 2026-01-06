# src/pipelines/zimage.py
from __future__ import annotations
from typing import Optional

import torch
from diffusers import ZImagePipeline


def load_zimage_turbo(
    model_id: str = "Tongyi-MAI/Z-Image-Turbo",
    device: Optional[str] = None,
    torch_dtype: Optional[torch.dtype] = None,
    use_safetensors: bool = True,
) -> ZImagePipeline:
    """
    Load Z-Image-Turbo pipeline for inference.
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    if torch_dtype is None:
        if device == "cuda":
            torch_dtype = torch.float16
        else:
            torch_dtype = torch.float32

    pipe = ZImagePipeline.from_pretrained(
        model_id,
        torch_dtype=torch_dtype,
        use_safetensors=use_safetensors,
    )
    pipe = pipe.to(device)

    if device == "cuda":
        try:
            pipe.enable_xformers_memory_efficient_attention()
        except Exception:
            pass

    return pipe