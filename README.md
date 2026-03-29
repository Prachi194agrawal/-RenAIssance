# Historical Spanish Manuscript OCR using Transformer & Vision-Language Models (2026)

<p align="center">
  <img src="https://img.shields.io/badge/GSoC-2026-orange?logo=google&logoColor=white"/>
  <img src="https://img.shields.io/badge/Organization-HumanAI-blue"/>
  <img src="https://img.shields.io/badge/Python-3.10+-blue?logo=python"/>
  <img src="https://img.shields.io/badge/Model-VLM%20%7C%20Transformer-green"/>
  <img src="https://img.shields.io/badge/Fine--tuning-LoRA-purple"/>
  <img src="https://img.shields.io/badge/License-MIT-lightgrey"/>
</p>

<p align="center">
  <b>CER: 0.628 → 0.485 (↓ 22.8%) &nbsp;|&nbsp; WER: 1.003 → 0.627 (↓ 37.5%)</b><br/>
  <i>Vision-Language Model + LoRA vs. Tesseract baseline on 17th-century Spanish documents</i>
</p>

---

##  Overview

Transcribing **17th-century Spanish printed and handwritten documents** is a challenging OCR problem due to:

* Archaic orthography and spelling
* Degraded document quality
* Historical glyph variations (e.g., long-ſ, u/v ambiguity)
* Non-standard layouts and abbreviations

Traditional OCR systems fail in this domain.

This project introduces a **hybrid transformer-based OCR pipeline** integrating:

* Vision Transformers (ViT)
* Transformer decoders
* Vision-Language Models (VLMs)
* Large Language Models (LLMs)

The system is designed to accurately transcribe **early modern Spanish texts** with strong contextual understanding.

---

## Project Goals

1. Develop a **hybrid end-to-end OCR model** using transformers
2. Improve transcription of **historical and degraded documents**
3. Integrate **LLM/VLM models** for contextual understanding
4. Expand dataset to include **printed + handwritten sources**
5. Achieve **≥ 90% transcription accuracy**

---

## Pipeline Overview

```
PDF Document / Manuscript Image
        │
        ▼
Rendering & Preprocessing
        │
        ▼
Data Augmentation
        │
        ▼
Transformer-Based OCR Model
        │
        ▼
LLM / VLM Post-processing
        │
        ▼
Final Transcription Output
```

---

##  Model Architecture

### Hybrid OCR System

* ViT + Transformer Decoder
* CNN + Transformer
* Vision-Language Model (Primary Approach)

### Core Flow

```
Input Image
   ↓
Vision Encoder
   ↓
Multimodal Projection
   ↓
Transformer / LLM Decoder
   ↓
Generated Text
```

### Fine-Tuning Strategy

* LoRA-based parameter-efficient tuning
* Applied to:

  * Attention layers
  * Feed-forward layers
  * Multimodal components

---

##  Model Performance

### Overall Results

| Metric    | Tesseract (Baseline) | Proposed Model | Improvement |
| --------- | -------------------- | -------------- | ----------- |
| **CER ↓** | 0.628                | **0.485**      | **↓ 22.8%** |
| **WER ↓** | 1.003                | **0.627**      | **↓ 37.5%** |

---

### Key Insights

* Strong improvement in **character-level accuracy**
* Better handling of **historical spelling patterns**
* Significant gains from **context-aware decoding**
* Minor drops only in **extreme augmentation cases**

---

## Dataset & Preprocessing

### Dataset

* 17th-century Spanish texts
* Printed + partially handwritten data
* Multiple typography styles

### Preprocessing

* PDF → Image conversion
* Resolution normalization
* Contrast enhancement
* Noise reduction
* JPEG compression

---

## Data Augmentation

* Rotation (±10°)
* Random cropping
* Contrast variation
* Multi-sample expansion

---

## Post-processing

* Context-aware correction using LLMs
* Historical grammar adaptation
* Character normalization:

  * u/v resolution
  * long-ſ conversion
  * abbreviation expansion

---

## Tasks Implemented

* Hybrid transformer OCR model
* Vision-language integration
* Context-aware correction
* Dataset expansion
* Evaluation (CER/WER)

---

## Quick Start (Run Directly)

###  Training Notebook

https://colab.research.google.com/drive/1GAVAzlilZEp4w0cvwrviRCPhHwOvWsY9?usp=sharing
 
### Evaluation Notebook

https://colab.research.google.com/drive/1LuC_b0QEr5DirhDEraxZQhCoUM8VchyR?usp=sharing

---

## Dataset Access

https://drive.google.com/drive/folders/1S1eC18EHBkXIJ4AJmLdT-fOzwxWvEWNC?usp=sharing

---

##  Evaluation Metrics

* CER (Character Error Rate)
* WER (Word Error Rate)



## Expected Outcomes

* Robust OCR for historical texts
* ≥ 90% accuracy
* Scalable ML pipeline
* Contribution to Digital Humanities

---

##  License

MIT License

---

##  Acknowledgements

* Google Summer of Code 2026
* HumanAI Initiative


---
