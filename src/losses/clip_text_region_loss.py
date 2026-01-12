from __future__ import annotations
from typing import List, Tuple
import torch
import torch.nn.functional as F
from transformers import CLIPProcessor, CLIPModel

class CLIPTextRegionLoss(torch.nn.Module):
    """
    Computes a cosine distance between CLIP(image_crop) and CLIP(text).
    This is differentiable w.r.t. the image crop pixels, hence can backprop into the generator.
    """

    def __init__(self, clip_id: str = "openai/clip-vit-base-patch32", device: str = "cuda"):
        super().__init__()
        self.device = device
        self.clip = CLIPModel.from_pretrained(clip_id).to(device)
        self.proc = CLIPProcessor.from_pretrained(clip_id)
        self.clip.eval()
        for p in self.clip.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def _encode_text(self, texts: List[str]) -> torch.Tensor:
        inputs = self.proc(text=texts, images=None, return_tensors="pt", padding=True, truncation=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items() if torch.is_tensor(v)}
        t = self.clip.get_text_features(**inputs)
        return F.normalize(t, dim=-1)

    def forward(self, image_batch: torch.Tensor, bboxes: torch.Tensor, texts: List[str]) -> torch.Tensor:
        """
        image_batch: (B,3,H,W) in [0,1]
        bboxes: (B,4) int xyxy in pixel coords
        """
        B, _, H, W = image_batch.shape
        assert bboxes.shape[0] == B

        # Crop each region, resize to 224x224 for CLIP
        crops = []
        for i in range(B):
            x1, y1, x2, y2 = bboxes[i].tolist()
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(W, x2), min(H, y2)
            crop = image_batch[i:i+1, :, y1:y2, x1:x2]
            crop = F.interpolate(crop, size=(224, 224), mode="bilinear", align_corners=False)
            crops.append(crop)
        crops = torch.cat(crops, dim=0)  # (B,3,224,224)

        # CLIP image features (need CLIP preprocessing normalization)
        # We'll approximate by using processor's normalization constants manually
        mean = torch.tensor([0.48145466, 0.4578275, 0.40821073], device=image_batch.device).view(1,3,1,1)
        std  = torch.tensor([0.26862954, 0.26130258, 0.27577711], device=image_batch.device).view(1,3,1,1)
        crops_norm = (crops - mean) / std

        img_feat = self.clip.get_image_features(pixel_values=crops_norm)
        img_feat = F.normalize(img_feat, dim=-1)

        # Text features (frozen)
        txt_feat = self._encode_text(texts)

        # cosine distance
        loss = 1.0 - (img_feat * txt_feat).sum(dim=-1)  # (B,)
        return loss.mean()
