from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Tuple, Any, List
import random

import numpy as np
from PIL import Image, ImageDraw, ImageFont
import torch
from torch.utils.data import Dataset

@dataclass
class SynthSample:
    image: Image.Image
    text: str
    bbox_xyxy: Tuple[int, int, int, int]  # (x1, y1, x2, y2)

def _rand_text(rng: random.Random) -> str:
    # Keep it simple and controllable; you can extend to Hebrew later.
    vocab = ["SALE", "OPEN", "HELLO", "POSTER", "EVENT", "FRIDAY", "SUMMER", "2026", "NIR", "DANA"]
    # random length 1-3 tokens
    k = rng.randint(1, 3)
    return " ".join(rng.sample(vocab, k=k))

def _get_font(size: int) -> ImageFont.ImageFont:
    # Uses PIL default font if no TTF available.
    try:
        return ImageFont.truetype("DejaVuSans.ttf", size=size)
    except Exception:
        return ImageFont.load_default()

def render_text_image(
    text: str,
    image_size: int = 512,
    rng: random.Random | None = None,
) -> SynthSample:
    rng = rng or random.Random()
    bg = (rng.randint(200, 255), rng.randint(200, 255), rng.randint(200, 255))
    img = Image.new("RGB", (image_size, image_size), bg)
    draw = ImageDraw.Draw(img)

    font_size = rng.randint(32, 72)
    font = _get_font(font_size)

    # Measure text
    bbox = draw.textbbox((0, 0), text, font=font)
    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]

    # Place text
    margin = 20
    x1 = rng.randint(margin, max(margin, image_size - tw - margin))
    y1 = rng.randint(margin, max(margin, image_size - th - margin))
    x2, y2 = x1 + tw, y1 + th

    # Optional box behind text for contrast
    if rng.random() < 0.7:
        pad = rng.randint(6, 16)
        rect = (x1 - pad, y1 - pad, x2 + pad, y2 + pad)
        rect = (
            max(0, rect[0]),
            max(0, rect[1]),
            min(image_size, rect[2]),
            min(image_size, rect[3]),
        )
        box_color = (rng.randint(0, 80), rng.randint(0, 80), rng.randint(0, 80))
        draw.rectangle(rect, fill=box_color)

    # Text color
    fg = (rng.randint(230, 255), rng.randint(230, 255), rng.randint(230, 255))
    draw.text((x1, y1), text, font=font, fill=fg)

    return SynthSample(image=img, text=text, bbox_xyxy=(x1, y1, x2, y2))

class SynthTextDataset(Dataset):
    def __init__(self, n: int, image_size: int = 512, seed: int = 42):
        self.n = n
        self.image_size = image_size
        self.seed = seed

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        rng = random.Random(self.seed + idx)
        text = _rand_text(rng)
        sample = render_text_image(text=text, image_size=self.image_size, rng=rng)

        # To tensor [0,1]
        x = torch.from_numpy(np.array(sample.image)).float() / 255.0
        x = x.permute(2, 0, 1)  # CHW

        return {
            "pixel_values": x,
            "text": sample.text,
            "bbox_xyxy": torch.tensor(sample.bbox_xyxy, dtype=torch.long),
        }

def collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    pixel_values = torch.stack([b["pixel_values"] for b in batch], dim=0)
    texts = [b["text"] for b in batch]
    bboxes = torch.stack([b["bbox_xyxy"] for b in batch], dim=0)
    return {"pixel_values": pixel_values, "texts": texts, "bboxes": bboxes}
