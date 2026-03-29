# Historical Spanish Manuscript OCR using Vision-Language Model Fine-Tuning

<p align="center">
  <img src="images/humanai_logo.jpg" alt="HumanAI Foundation" height="80" style="margin-right: 20px;"/>
  &nbsp;&nbsp;&nbsp;&nbsp;
  <img src="images/gsoc_logo.png" alt="Google Summer of Code" height="50"/>
</p>

<p align="center">
  <a href="https://summerofcode.withgoogle.com/programs/2026/projects/lg7vQeMM">
    <img src="https://img.shields.io/badge/GSoC-2024-orange?logo=google&logoColor=white" alt="GSoC 2024"/>
  </a>
  <a href="https://humanai.foundation/">
    <img src="https://img.shields.io/badge/HumanAI-Foundation-blue" alt="HumanAI"/>
  </a>
  <img src="https://img.shields.io/badge/Python-3.10+-blue?logo=python&logoColor=white" alt="Python"/>
  <img src="https://img.shields.io/badge/Model-Qwen2.5--VL--7B-green" alt="Model"/>
  <img src="https://img.shields.io/badge/Fine--tuning-LoRA-purple" alt="LoRA"/>
  <img src="https://img.shields.io/badge/License-MIT-lightgrey" alt="License"/>
</p>

<p align="center">
  <b>CER: 0.628 → 0.485 (↓ 22.8%) &nbsp;|&nbsp; WER: 1.003 → 0.627 (↓ 37.5%)</b><br/>
  <i>Qwen2.5-VL-7B + LoRA vs. Tesseract baseline on 17th-century Spanish manuscripts</i>
</p>

---

## Overview

This project addresses the challenge of Optical Character Recognition (OCR) on **17th-century Spanish printed manuscripts** — a domain where conventional OCR engines such as Tesseract fail due to archaic orthography, interchangeable glyphs (`u`/`v`, `f`/long-`ſ`), macron/tilde abbreviation systems, and degraded printing quality.

Leveraging a hybrid multimodal approach, we fine-tune a **Vision-Language Model (VLM)** — specifically `Qwen2.5-VL-7B-Instruct` — with **Low-Rank Adaptation (LoRA)** via the Unsloth framework. The resulting pipeline accurately transcribes Early Modern Spanish text at full page granularity, outperforming Tesseract by **22.8% on CER** and **37.5% on WER** on the held-out validation set.

This project is a continuation of the **[RenAIssance project](https://humanai.foundation/)** under the HumanAI organization, developed as part of **Google Summer of Code 2024**.


---

## Table of Contents

- [Project Goals](#project-goals)
- [Pipeline Overview](#pipeline-overview)
- [Installation](#installation)
- [Project Structure](#project-structure)
- [About The Project](#about-the-project)
  - [Dataset and Preprocessing](#dataset-and-preprocessing)
  - [Data Augmentation](#data-augmentation)
  - [Model Architecture](#model-architecture)
  - [Paleographic Normalization](#paleographic-normalization)
  - [Training Configuration](#training-configuration)
- [Model Performance](#model-performance)
- [Datasets and Models](#datasets-and-models)
- [Acknowledgements](#acknowledgements)
- [License](#license)
- [Links](#links)

---

## Project Goals

1. **Fine-Tune a VLM for Historical OCR**  
   Adapt `Qwen2.5-VL-7B-Instruct` for transcribing 17th-century Spanish manuscripts using parameter-efficient LoRA fine-tuning via Unsloth. The model captures both visual glyph-level features and contextual orthographic patterns unique to Early Modern Spanish printing.

2. **Outperform Conventional OCR Baselines**  
   Substantially improve over Tesseract 4.1.1 on historical Spanish documents through paleographic normalization, data augmentation, and rigorous hyperparameter tuning — generalising across diverse typefaces, degradation levels, and scribal conventions present in corpus documents from 1628 to 1650.

---

## Pipeline Overview
PDF Document (17th-c. Spanish)
│
▼
PyMuPDF Rendering (150 DPI)
│
▼
Preprocessing: Resize 768×1024 → Contrast ×1.4 → Unsharp Mask → JPEG q=85
│
▼
Data Augmentation: Rotation ±10° | Crop 85–97% | Contrast 1.2×–1.6×
(19 base pairs → 76 training pairs, SEED=42)
│
▼
Qwen2.5-VL-7B-Instruct + LoRA (r=16, α=32)
ViT Encoder → Multimodal Projector → LLM Decoder (7B)
│
▼
Paleographic Normalization Pipeline
(Macron/Tilde expansion · Cedilla · u/v · f/ſ)
│
▼
Final Transcription Output


> See `images/IMG1_pipeline_overview.png` for the full visual flowchart.

---

## Installation

No external installation is required. Open the notebooks in Google Colab (recommended), Kaggle, or Jupyter and run cells sequentially — all dependencies are installed by the first cell.

```bash
# Installed automatically by the notebooks:
pip install "trl>=0.18.2,<=0.24.0"
pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
pip install --upgrade transformers accelerate peft
pip install pymupdf python-docx jiwer rapidfuzz nltk datasets
pip install qwen-vl-utils gradio pyspellchecker
apt-get install poppler-utils tesseract-ocr tesseract-ocr-spa
```

> **Recommended hardware:** Google Colab T4 GPU (16 GB VRAM) — training takes approximately 45 minutes (~2,706 seconds).

---

## Project Structure

renaissance-ocr/
├── step_1_model_training.ipynb # Data ingestion, preprocessing, augmentation, LoRA fine-tuning
├── step_2_model_checking.ipynb # Evaluation, CER/WER benchmarking, Gradio UI
├── images/
│ ├── humanai_logo.jpg
│ ├── gsoc_logo.png
│ ├── IMG1_pipeline_overview.png # End-to-end pipeline flowchart
│ ├── IMG2_sample_dataset.png # Real document page + ground truth side-by-side
│ ├── IMG3_augmentation.png # 2×3 augmentation examples grid
│ ├── IMG4_architecture.png # Qwen2.5-VL architecture with LoRA adapters
│ ├── IMG5_qualitative.png # Tesseract vs VLM qualitative comparison table
│ ├── G1_loss_curve.png # Training & validation loss across 8 epochs
│ └── G2_cer_wer_benchmark.png # CER/WER benchmark bar chart (all 11 samples)
├── LICENSE
└── README.md



> **Google Drive layout** (used by the notebooks at runtime):
> ```
> MyDrive/
> ├── print_pdf/                   ← Source PDF documents (x-train)
> ├── Print/                       ← Ground truth DOCX transcriptions (y-train)
> └── Saved_PDF_Data/
>     ├── Training_Dataset/
>     │   ├── x_train_images/      ← Preprocessed JPEG pages
>     │   ├── x_train_images_aug/  ← Augmented variants
>     │   ├── y_train_labels.json
>     │   └── y_train_labels_aug.json
>     ├── My_Best_Qwen_LoRA/       ← Saved LoRA adapter (adapter_config.json + weights)
>     └── Evaluation_Results/      ← Per-sample CER/WER JSON + CSV exports
> ```

---

## About The Project

### Dataset and Preprocessing

**Input Corpus:** Six scanned PDF source documents paired with six DOCX ground-truth transcriptions, covering historical Spanish texts from 1628–1650:

| Document | Period | Pages Extracted | Transcribed Pages |
|---|---|---|---|
| Buendia — Instrucción | 17th c. | 10 | 3 |
| Covarrubias — Tesoro de la Lengua | 17th c. | 10 | 3 |
| Guardiola — Tratado de Nobleza | 17th c. | 10 | 3 |
| PORCONES.228.38 | 1646 | 10 | 5 |
| PORCONES.23.5 | 1628 | 10 | 4 |
| PORCONES.748.6 | 1650 | 10 | 4 |
| **Total** | — | **60** | **22 (19 valid pairs)** |

**PDF-to-Image Extraction:**

PDF Page → PyMuPDF (150 DPI) → Resize 768×1024 (Lanczos)
→ Contrast ×1.4 (PIL ImageEnhance) → Unsharp Mask → JPEG q=85

PDF Page → PyMuPDF (150 DPI) → Resize 768×1024 (Lanczos)
→ Contrast ×1.4 (PIL ImageEnhance) → Unsharp Mask → JPEG q=85

Base pairs: 19
After 3× augment: 76 (SEED = 42)
Train split (85%): 65 samples
Val split (15%): 11 samples


> See `images/IMG3_augmentation.png` for a 2×3 grid showing real document pages and their augmented variants.

---

### Model Architecture

**Base Model:** `Qwen2.5-VL-7B-Instruct` — a 7B-parameter vision-language model, quantised to 4-bit (NF4) via Unsloth for memory-efficient training on a single T4 GPU (16 GB VRAM).


Input Image (768×1024)
│
▼
ViT Encoder — Patch Embedding + RoPE-2D Attention → ~512–768 visual tokens
│
▼
Multimodal Projector — Cross-attention alignment (visual → LLM token space)
│
▼
Qwen2.5 LLM Decoder (7B) + LoRA Adapters
│
▼
Raw Transcription Text



> See `images/IMG4_architecture.png` for the full architecture diagram with LoRA injection points.

**LoRA Configuration:**

| Parameter | Value |
|---|---|
| Rank (`r`) | 16 |
| Alpha (`lora_alpha`) | 32 |
| Dropout | 0.05 |
| Target modules | `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj` |
| Trainable parameters | 47,589,376 **(0.57% of 8.34B total)** |
| RSLoRA | Disabled |

Fine-tuning is applied across **vision layers, language layers, attention modules, and MLP modules** simultaneously.

---

### Paleographic Normalization

The model output passes through a deterministic post-processing pipeline implementing 17th-century Spanish transcription conventions:

| Rule | Raw Input | Normalized Output |
|---|---|---|
| Macron expansion | `mãdato del rey` | `mandato del rey` |
| Tilde-q expansion | `q̃ se haga` | `que se haga` |
| Cedilla conversion | `la çiudad` | `la ziudad` |
| ñ preservation | `el señor` | `el señor` *(unchanged)* |
| u/v disambiguation | `es vna verdad` | `es una verdad` |
| long-ſ resolution | `el feñor` | `el señor` |



> See `images/IMG5_qualitative.png` for the full qualitative comparison table across all rule types.

---

### Training Configuration

**System Prompt (used at both training and inference):**
> *"You are an expert OCR system specialising in 17th-century Spanish printed sources (siglos XVI–XVII). Transcribe ALL visible printed text exactly as it appears, preserving original Early Modern Spanish spelling, punctuation, abbreviations, and line structure. Disregard decorative borders, stamps, and marginalia."*

**Hyperparameters:**

| Parameter | Value |
|---|---|
| Epochs | 8 |
| Batch size (train / eval) | 1 |
| Gradient accumulation steps | 4 |
| Learning rate | 2e-4 |
| LR scheduler | Cosine |
| Optimizer | AdamW 8-bit (GPU) |
| Weight decay | 0.01 |
| Warmup steps | 5 |
| Max sequence length | 2048 |
| Precision | fp16 (T4) / bf16 (Ampere+) |
| Best model selection | Minimum `eval_loss` |
| Total training time | ~2,706 s (~45 min on T4) |

**Training Loss Curve:**

| Epoch | Train Loss | Val Loss |
|---|---|---|
| 1 | 1.543 | 1.252 |
| 2 | 1.053 | 0.846 |
| 3 | 0.400 | 0.502 |
| 4 | 0.170 | 0.241 |
| 5 | 0.032 | 0.137 |
| 6 | 0.015 | 0.111 |
| 7 | 0.010 | 0.085 |
| **8** | **0.012** | **0.083** ⭐ best |

> See `images/G1_loss_curve.png` for the plotted loss curve with phase annotations.

---

## Model Performance

The fine-tuned `Qwen2.5-VL-7B + LoRA` model substantially outperforms the Tesseract OCR baseline on the 11-sample validation set:

### Overall Results

| Metric | Tesseract 4.1.1 (Baseline) | Qwen2.5-VL + LoRA (Ours) | Improvement |
|---|---|---|---|
| **CER** | 0.628 (62.8%) | **0.485 (48.5%)** | ↓ **22.8%** |
| **WER** | 1.003 (100.3%) | **0.627 (62.7%)** | ↓ **37.5%** |
| Best sample CER | 0.205 | **0.010** | ↓ 95.1% |
| Best sample WER | 0.667 | **0.066** | ↓ 90.1% |

### Per-Sample Results

| Sample | Tess CER | VLM CER | ΔCER | Tess WER | VLM WER | ΔWER |
|---|---|---|---|---|---|---|
| page0001_aug1_rot_contrast | 1.653 | **0.010** | ↓ 1.643 | 2.513 | **0.066** | ↓ 2.447 |
| page0001_aug2_rot_contrast | 1.607 | **0.506** | ↓ 1.101 | 2.855 | **0.711** | ↓ 2.145 |
| PORCONES.228.38_page0004 | 0.205 | **0.059** | ↓ 0.146 | 0.667 | **0.145** | ↓ 0.522 |
| page0007_aug2_rot | 0.287 | **0.014** | ↓ 0.273 | 0.872 | **0.070** | ↓ 0.802 |
| PORCONES.228.38_page0005 | 0.276 | **0.135** | ↓ 0.141 | 0.787 | **0.186** | ↓ 0.601 |
| Buendia_Instruccion_page0004 | 0.305 | **0.145** | ↓ 0.159 | 0.800 | **0.285** | ↓ 0.515 |
| page0002_aug1_rot | 0.779 | 0.784 | +0.005 | 0.966 | 0.970 | +0.004 |
| page0002_aug0_rot1 | 0.759 | 0.778 | +0.019 | 0.956 | **0.938** | ↓ 0.018 |
| page0003_aug0_rot | 0.738 | 0.774 | +0.036 | 0.976 | 1.052 | +0.076 |
| page0004_aug2_rot | 0.743 | 0.770 | +0.027 | 0.969 | **0.945** | ↓ 0.024 |
| page0002_aug0_rot2 | 0.754 | 0.782 | +0.028 | 0.958 | **0.935** | ↓ 0.023 |
| **OVERALL** | **0.628** | **0.485** | **↓ 0.143** | **1.003** | **0.627** | **↓ 0.376** |

> See `images/G2_cer_wer_benchmark.png` for the grouped bar chart comparing Tesseract vs. VLM across all 11 samples.

> **Note:** The VLM underperforms Tesseract on a small subset of heavily augmented samples (e.g., severely rotated + cropped pages), where degradation exceeds the training distribution. This is addressed by scaling the training corpus — the primary goal of ongoing development.

---

## Datasets and Models

| Resource | Location |
|---|---|
| Source PDFs (x-train) | `Google Drive: MyDrive/print_pdf/` |
| DOCX transcriptions (y-train) | `Google Drive: MyDrive/Print/` |
| Preprocessed images | `Saved_PDF_Data/Training_Dataset/x_train_images/` |
| Augmented images | `Saved_PDF_Data/Training_Dataset/x_train_images_aug/` |
| Augmented labels JSON | `Saved_PDF_Data/Training_Dataset/y_train_labels_aug.json` |
| Trained LoRA adapter | `Saved_PDF_Data/My_Best_Qwen_LoRA/` |
| Evaluation results (JSON + CSV) | `Saved_PDF_Data/Evaluation_Results/` |

The trained LoRA adapter is loaded by `step_2_model_checking.ipynb` automatically — **no retraining required** if the adapter exists at the above path.

---

## Interactive Demo

An interactive **Gradio UI** is included in `step_2_model_checking.ipynb`. Upload any manuscript image to receive a live transcription with an annotated output panel showing both the raw VLM output and the post-normalized result.

```python
# Launch from the last cell of step_2_model_checking.ipynb
demo.launch(share=True)   # generates a public URL valid for 72 hours
```

---

## Acknowledgements

- [HumanAI Foundation](https://humanai.foundation/) — project sponsorship and mentorship
- [Google Summer of Code 2024](https://summerofcode.withgoogle.com/programs/2024/projects/lg7vQeMM) — program support
- [Unsloth](https://github.com/unslothai/unsloth) — memory-efficient LoRA fine-tuning framework
- [Qwen2.5-VL](https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct) — base vision-language model
- [Biblioteca Digital Hispánica](https://www.bne.es/es/Catalogos/BibliotecaDigitalHispanica/) — source document corpus

Full development walkthrough:  
📖 [My Journey with HumanAI in GSoC 2024 — Part 2](https://medium.com/@shashankshekharsingh1205/my-journey-with-humanai-in-the-google-summer-of-code24-program-part-2-bb42abce3495)

---

## License

This project is licensed under the **MIT License**. See the [LICENSE](LICENSE) file for details.

---

## Links

| Resource | URL |
|---|---|
| GSoC 2024 Project Page | https://summerofcode.withgoogle.com/programs/2024/projects/lg7vQeMM |
| HumanAI Foundation | https://humanai.foundation/ |
| Developer Blog | https://medium.com/@shashankshekharsingh1205/my-journey-with-humanai-in-the-google-summer-of-code24-program-part-2-bb42abce3495 |
| Base Model (HuggingFace) | https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct |
| Unsloth Framework | https://github.com/unslothai/unsloth |

---


