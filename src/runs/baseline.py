from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional
import json

import torch

from src.pipelines.zimage import load_zimage_turbo
from src.utils.seed import set_seed
from src.utils.io import ensure_dir, read_jsonl, write_json, write_jsonl, timestamp


def generate_baseline(
    prompts_path: str = "data/prompts/text_rendering_prompts.jsonl",
    out_root: str = "experiments/runs",
    run_name: Optional[str] = None,
    model_id: str = "Tongyi-MAI/Z-Image-Turbo",
    seed: int = 42,
    num_inference_steps: int = 8,
    guidance_scale: float = 0.0,
    height: int = 1024,
    width: int = 1024,
    device: Optional[str] = None,
    dtype: str = "fp16",  # fp16/bf16/fp32
) -> Path:
    """
    Generate baseline samples and save:
    - images/*.png
    - metadata.json
    - prompts.jsonl (copy)
    - outputs.jsonl (per-sample)
    """
    set_seed(seed)

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    dtype_map = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}
    torch_dtype = dtype_map.get(dtype, torch.float16)

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
        "num_prompts": len(prompts),
        "prompts_file": prompts_path,
    }
    write_json(run_dir / "metadata.json", meta)
    write_jsonl(run_dir / "prompts.jsonl", prompts)

    pipe = load_zimage_turbo(
        model_id=model_id,
        device=device,
        torch_dtype=torch_dtype,
    )

    outputs_path = run_dir / "outputs.jsonl"
    if outputs_path.exists():
        outputs_path.unlink()

    for i, item in enumerate(prompts, start=1):
        prompt = item["prompt"]

        gen = torch.Generator(device=pipe.device).manual_seed(seed + i)
        out = pipe(
            prompt=prompt,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            height=height,
            width=width,
            generator=gen,
        )
        img = out.images[0]

        img_name = f"{i:04d}.png"
        img.save(images_dir / img_name)

        rec = dict(item)
        rec["index"] = i
        rec["sample_seed"] = seed + i
        rec["image_path"] = str(Path("images") / img_name)

        with open(outputs_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

        print(f"[{i}/{len(prompts)}] saved {img_name}")

    print(f"\nDone. Run saved to: {run_dir}")
    return run_dir
