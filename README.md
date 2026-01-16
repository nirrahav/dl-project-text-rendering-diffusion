# Auxiliary Loss for Better Text Rendering (Z-Image)

A small research / course project that explores improving text rendering from diffusion models
by fine-tuning with a CLIP-based auxiliary loss applied to text regions.

This repository contains code to synthesize simple text images, compute a CLIP text-region loss,
and fine-tune a diffusion model on synthetic data for quick experiments.

**Quick Links**

- Notebook: `colab/Final_project.ipynb`
- Main code: `src/`

**Requirements**

- Python 3.9+ (see `requirements.txt`)
- GPU recommended for training experiments

**Quickstart (local)**

1. Create environment and install dependencies:

   ```bash
   python -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   ```

2. Inspect or run training script (examples under `src/train`), or open the Colab notebook.

**Run in Colab**

- Open `colab/Final_project.ipynb` in Google Colab. The notebook clones this repository,
  installs dependencies, and runs a small-scale fine-tuning and evaluation pipeline.

**Repository Structure**

- `colab/` — Colab notebook used for experiments
- `src/` — project source code
  - `src/data` — synthetic dataset generation
  - `src/train` — training scripts and finetuning logic
  - `src/losses` — CLIP text-region loss implementation
  - `src/utils` — helper utilities

**Notes & Caveats**

- The experiments are intentionally small-scale for fast iteration and reproduceability.
- Full-scale Z-Image training is out of scope for this repo.

**Contributing / Contact**

- Feel free to open issues or PRs for bug fixes and improvements.

---

*Generated: improved README for clarity and quick usage.*

