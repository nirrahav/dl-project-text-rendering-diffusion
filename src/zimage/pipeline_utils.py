from __future__ import annotations
import torch
from diffusers import ZImagePipeline

def load_zimage(model_id: str, dtype: str = "bf16", device: str = "cuda") -> ZImagePipeline:
    if dtype == "bf16":
        torch_dtype = torch.bfloat16
    elif dtype == "fp16":
        torch_dtype = torch.float16
    else:
        torch_dtype = torch.float32

    pipe = ZImagePipeline.from_pretrained(
        model_id,
        torch_dtype=torch_dtype,
        low_cpu_mem_usage=False,
    )
    pipe = pipe.to(device)
    return pipe
