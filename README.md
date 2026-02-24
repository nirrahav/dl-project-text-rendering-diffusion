# Improving Text Rendering in Diffusion Models via Auxiliary Loss  
### Deep Learning Final Project – BGU IEM 2026  

**Authors:**  
Nir Rahav 
Dana Benaim

---

## 📌 Overview

Diffusion-based text-to-image models generate visually coherent images from textual prompts.  
However, text rendering inside generated images remains a persistent challenge.

In most diffusion models, textual content emerges as a byproduct of the image generation process and is not explicitly optimized. As a result, generated text is often distorted, misspelled, or geometrically inconsistent.

This project investigates whether introducing an **auxiliary text-aware loss** during fine-tuning can improve text rendering quality without modifying the original model architecture.

---

## 🧠 Motivation

Text rendering is inherently difficult for diffusion models because:

- Image generation is continuous and probabilistic.
- Text is discrete and structurally constrained.
- Small pixel deviations can change character identity.
- High CLIP similarity does not guarantee readable text.

We aim to bridge this gap using targeted objective-level supervision.

---

## 🏗 Base Model

We fine-tune the publicly available:

> **Tongyi-MAI Z-Image-Turbo**

Z-Image-Turbo is a diffusion-based foundation model built on a Scalable Single-Stream Diffusion Transformer (S3-DiT).  
It demonstrates strong bilingual capabilities but does not explicitly optimize for character-level textual correctness.

---

## 🔬 Method

We introduce an auxiliary loss term focused on improving textual rendering quality.

The training objective is modified as:

L_total = L_diffusion + λ * L_aux

Where:

- `L_diffusion` – standard diffusion noise prediction loss  
- `L_aux` – text-aware auxiliary supervision  
- `λ = 0.2` – weighting coefficient  

Key design principles:

- ✅ No architectural changes  
- ✅ No inference overhead  
- ✅ Preserves original 8-step inference  
- ✅ Fine-tuning at 512×512 resolution  

---

## ⚙️ Training Setup

- Resolution: 512×512  
- Auxiliary loss weight: λ = 0.2  
- Controlled fine-tuning on pretrained Z-Image-Turbo  
- Gradient accumulation used to simulate larger effective batch size under limited computational resources  
- Evaluation performed on a controlled prompt set  

---

## 📊 Evaluation Metrics

We evaluate both quantitatively and qualitatively.

### Quantitative Metrics

- **CLIP Score** – semantic alignment between prompt and image  
- **CER (Character Error Rate)** – OCR-based character-level error  
- **Exact Match** – full string correctness  

| Metric | Baseline | Fine-Tuned | Δ |
|--------|----------|------------|---|
| CLIP Score | 0.3629 | 0.3631 | +0.0002 |
| CER | 0.9955 | 1.0013 | +0.0058 |
| Exact Match | 0.0000 | 0.0000 | 0.0000 |

While semantic alignment remained stable, no measurable improvement was observed in OCR-based metrics.

---

### 👁 Qualitative Analysis

Despite limited quantitative gains, qualitative inspection reveals:

- Improved stroke consistency  
- Sharper letter edges  
- More stable geometric structure  
- Better text alignment in structured layouts  

However:

- Character-level distortions still appear  
- Spacing inconsistencies remain  
- Thin fonts and digital-style displays remain sensitive to artifacts  

This highlights the discrepancy between automated metrics and human perception.

---

## 🎯 Contributions

- Introduced a text-aware auxiliary loss integrated at the objective level  
- Demonstrated that controlled fine-tuning can improve typographic stability  
- Preserved original architecture and inference configuration  
- Showed that visual improvements do not necessarily translate to symbolic correctness  

---

## 🚧 Limitations

- No measurable improvement in CER or Exact Match  
- Improvements primarily aesthetic rather than symbolic  
- Limited computational budget  
- Evaluation lacks systematic human study  

---

## 🔮 Future Work

- Incorporate structured human evaluation  
- Refine loss formulation for stronger character-level supervision  
- Explore alternative training configurations  
- Extend support to additional languages  
- Investigate multilingual and complex writing systems  

---

## 🏁 Final Remarks

This project demonstrates that objective-level supervision can improve perceptual text quality in diffusion models, while also revealing the challenges of enforcing symbolic correctness in continuous generative frameworks.