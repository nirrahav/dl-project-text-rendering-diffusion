# Improving Text Rendering in Diffusion Models via Auxiliary Loss

This project explores a fine-tuning approach for improving text rendering quality in diffusion-based image generation models.

We build on top of **Z-Image-Turbo**, a fast diffusion model optimized for few-step inference, and introduce a **text-aware auxiliary loss** that explicitly guides the model to generate clearer and more consistent text inside images.

The key idea is to improve text readability **without modifying the original model architecture**, by extending the training objective.

---

## Project Motivation

Modern text-to-image diffusion models produce visually impressive images, but often struggle when rendering text inside images. Common issues include distorted characters, inconsistent spacing, and unreadable words. These errors are especially problematic because small visual inaccuracies can lead to large semantic mistakes.

The problem becomes even more pronounced when moving beyond English to additional languages.

This project addresses these limitations by introducing an explicit training signal that focuses on text quality.

---

## Method Overview

The proposed approach extends the original training objective of Z-Image-Turbo
by adding an auxiliary, text-aware loss. The original diffusion loss is kept
unchanged, and the auxiliary loss is applied during fine-tuning.

The combined objective is defined as:

L_total = L_diffusion + λ · L_aux

Where:
- **L_diffusion** is the original diffusion training loss  
- **L_aux** is the text-aware auxiliary loss  
- **λ** controls the influence of the auxiliary loss

---

## Training Pipeline

The fine-tuning process follows these steps:

1. Encode prompts and images using the pretrained pipeline  
2. Perform a diffusion forward pass  
3. Predict latent representations  
4. Decode latents into images  
5. Apply the auxiliary loss on text regions  
6. Backpropagate the combined loss  

No additional sampling steps or architectural changes are introduced during training.

---

## Experimental Setup

- **Model:** Z-Image-Turbo (`Tongyi-MAI/Z-Image-Turbo`)  
- **Image Resolution:** 512×512  
- **Batch Size:** 1 (with gradient accumulation)  
- **Precision:** bfloat16  
- **Learning Rate:** 1e-5  
- **Auxiliary Loss Weight (λ):** 0.2  
- **Auxiliary Loss Frequency:** every 8 steps  
- **Training Type:** Fine-tuning only  

---

## Results

Qualitative comparisons between the baseline and fine-tuned models demonstrate clear improvements in text rendering quality.

Observed improvements include:
- Sharper and more readable text
- Improved alignment and spacing
- More consistent character thickness
- Reduced visual artifacts around letters

All comparisons are performed using identical prompts and inference settings, isolating the effect of the auxiliary loss.

---

## Usage

### Training

Fine-tuning is performed using the configuration defined in `TrainConfig` and executed from the Colab notebook.  
The training process updates only selected components of the model while preserving the original architecture.

### Inference and Evaluation

The notebook demonstrates how to:
- Generate baseline images using the pretrained model
- Load fine-tuned transformer weights
- Generate fine-tuned images
- Perform a side-by-side qualitative comparison

---

## Authors

- **Nir Rahav**
- **Dana Benaim**

Course Project – Deep Learning

---

## Notes and Future Work

This project focuses on qualitative improvements in text rendering. Possible future extensions include:
- OCR-based quantitative evaluation (e.g., CER/WER)
- Extension to additional languages
- Adaptive weighting of the auxiliary loss
- Application to other diffusion-based architectures

---

## Acknowledgments

- **Z-Image-Turbo** by Tongyi-MAI  
- **Hugging Face Diffusers** library  
