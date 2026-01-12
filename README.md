# Auxiliary Loss for Better Text Rendering (Z-Image)

This repo implements a small fine-tuning experiment that augments diffusion training with an auxiliary
text-region loss (CLIP-based) on a synthetic dataset (rendered text + known bbox).

- Python code lives under `src/`
- Colab notebook clones this repo, installs deps, runs training + inference comparisons.

Notes:
- Full Z-Image training is not feasible for a course project.
- We fine-tune with small batches on synthetic text images.
