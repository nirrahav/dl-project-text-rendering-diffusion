from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import json

import torch
import numpy as np

from src.pipelines.zimage import load_zimage_turbo
from src.utils.seed import set_seed
from src.utils.io import ensure_dir, read_jsonl, write_json, write_jsonl, timestamp


def _dtype_from_str(dtype: str) -> torch.dtype:
    dtype_map = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}
    if dtype not in dtype_map:
        raise ValueError(f"Unsupported dtype={dtype}. Choose from {list(dtype_map)}")
    return dtype_map[dtype]


def _is_black_or_invalid(img) -> bool:
    """
    Detect obviously broken outputs:
    - all/mostly black image
    - NaNs after conversion
    """
    arr = np.asarray(img)
    if arr.size == 0:
        return True
    if np.isnan(arr).any():
        return True
    # if max is 0 => fully black; if max very low => near-black
    return arr.max() <= 2


def _make_generator(device: str, seed: int) -> torch.Generator:
    # diffusers expects generator device to match compute device ("cuda" usually)
    if device.startswith("cuda"):
        return torch.Generator(device="cuda").manual_seed(seed)
    return torch.Generator(device="cpu").manual_seed(seed)


def generate_baseline(
    prompts_path: str = "data/prompts/text_rendering_prompts.jsonl",
    out_root: str = "experiments/runs",
    run_name: Optional[str] = None,
    model_id: str = "Tongyi-MAI/Z-Image-Turbo",
    seed: int = 42,
    num_inference_steps: int = 8,
    guidance_scale: float = 0.0,
    height: int = 512,   # ✅ safer default for baseline
    width: int = 512,    # ✅ safer default for baseline
    device: Optional[str] = None,
    dtype: str = "fp16",  # fp16/bf16/fp32
    fallback_dtypes: Tuple[str, ...] = ("bf16", "fp32"),  # ✅ auto-retry if black
) -> Path:
    """
    Generate baseline samples and save:
    - images/*.png
    - metadata.json
    - prompts.jsonl (copy)
    - outputs.jsonl (per-sample)

    Improvements vs v1:
    - VAE decoding stability assumed in loader (VAE fp32)
    - generator is created explicitly on correct device
    - detects black outputs and retries with safer precision (bf16/fp32)
    - safer default resolution (512) to validate baseline first
    """
    set_seed(seed)

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    requested_dtype = _dtype_from_str(dtype)

    run_name = run_name or f"{timestamp()}_baseline_zimage_turbo"
    run_dir = ensure_dir(Path(out_root) / run_name)
    images_dir = ensure_dir(run_dir / "images")

    prompts: List[Dict[str, Any]] = read_jsonl(prompts_path)

    meta = {
        "run_name": run_name,
        "model_id": model_id,
        "seed": seed,
        "num_inference_steps": num_inference_steps,
        "guidance_scale": guidance_scale,
        "height": height,
        "width": width,
        "device": device,
        "dtype": dtype,
        "fallback_dtypes": list(fallback_dtypes),
        "num_prompts": len(prompts),
        "prompts_file": prompts_path,
    }
    write_json(run_dir / "metadata.json", meta)
    write_jsonl(run_dir / "prompts.jsonl", prompts)

    # Load pipeline once with requested dtype
    pipe = load_zimage_turbo(
        model_id=model_id,
        device=device,
        torch_dtype=requested_dtype,
    )

    # Debug prints (very helpful in Colab)
    try:
        vae_dtype = next(pipe.vae.parameters()).dtype
        vae_dev = next(pipe.vae.parameters()).device
    except Exception:
        vae_dtype, vae_dev = None, None
    print(f"[pipeline] device={pipe.device} requested_dtype={requested_dtype} vae_dtype={vae_dtype} vae_device={vae_dev}")

    outputs_path = run_dir / "outputs.jsonl"
    if outputs_path.exists():
        outputs_path.unlink()

    for i, item in enumerate(prompts, start=1):
        prompt = item["prompt"]
        sample_seed = seed + i

        # Try requested dtype first, then fallbacks if output is black
        tried = [dtype] + [d for d in fallback_dtypes if d != dtype]
        img = None
        used_dtype = None

        for attempt_dtype in tried:
            # If dtype changes, re-load pipeline (simpler + stable)
            if attempt_dtype != dtype:
                pipe = load_zimage_turbo(
                    model_id=model_id,
                    device=device,
                    torch_dtype=_dtype_from_str(attempt_dtype),
                )

            gen = _make_generator(str(pipe.device), sample_seed)

            out = pipe(
                prompt=prompt,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                height=height,
                width=width,
                generator=gen,
            )
            candidate = out.images[0]

            if not _is_black_or_invalid(candidate):
                img = candidate
                used_dtype = attempt_dtype
                break

        if img is None:
            # Save the last candidate anyway for debugging
            img = candidate
            used_dtype = tried[-1]

        img_name = f"{i:04d}.png"
        img.save(images_dir / img_name)

        rec = dict(item)
        rec["index"] = i
        rec["sample_seed"] = sample_seed
        rec["image_path"] = str(Path("images") / img_name)
        rec["used_dtype"] = used_dtype

        with open(outputs_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

        print(f"[{i}/{len(prompts)}] saved {img_name} (dtype={used_dtype})")

    print(f"\nDone. Run saved to: {run_dir}")
    return run_dir