# 📜 17th-Century Spanish OCR using Transformers + LLM Post-Processing

## 🚀 Overview
This project addresses Optical Character Recognition (OCR) for **17th-century Spanish printed texts**, a challenging domain where traditional OCR systems (e.g., Tesseract) perform poorly due to:
- Irregular typography
- Degraded scans
- Historical spelling variations

The solution implements a **transformer-based OCR pipeline** combined with **LLM/VLM-based post-processing** to improve transcription accuracy.

> 📌 This work is submitted for **RenAIssance GSoC 2026 – Evaluation Test I (Printed OCR)**

---

## 🧠 Methodology

### 🔹 1. OCR Model (Transformer-Based)
- Built a deep learning OCR pipeline using transformer-based architecture
- Processes scanned document images to extract text
- Focused on:
  - Capturing **main textual content**
  - Ignoring **marginalia and decorative elements**

### 🔹 2. LLM/VLM Post-Processing (Late Stage)
- Applied after OCR inference
- Corrects:
  - OCR errors (missing/incorrect characters)
  - Word-level inconsistencies
  - Contextual spelling using historical patterns

---

## 📊 Results

### 🔸 Evaluation Metrics
- **CER (Character Error Rate)**
- **WER (Word Error Rate)**

### 🔸 Performance Improvement
| Metric | Baseline (Tesseract) | Proposed Pipeline | Improvement |
|--------|----------------------|------------------|------------|
| CER    | Higher               | Lower            | ↓ 22.8%    |
| WER    | Higher               | Lower            | ↓ 37.5%    |

### 🔸 Key Observations
- LLM/VLM significantly improves OCR output quality
- Strong robustness across different page layouts
- Effective handling of noisy historical scans

---

## 📁 Project Structure
├── step_1_model_training.ipynb # OCR model training pipeline
├── step_2_model_checking.ipynb # Evaluation & benchmarking
├── results.png # CER/WER comparison visualization
├── dataset/ # Input data (PDFs / images)
└── README.md




---

## 📦 Dataset

- Early modern Spanish printed documents (provided dataset)
- Ground truth transcriptions used for evaluation
- Preprocessing steps:
  - PDF → image conversion
  - Region focus on main text

---

## ⚙️ Installation & Usage

### 1. Install dependencies

pip install -r requirements.txt

### 2. Train OCR Model

jupyter notebook step_1_model_training.ipynb

### Run Evaluation

jupyter notebook step_2_model_checking.ipynb

### Evaluation Details
Metrics Used:
CER (Character Error Rate)
Measures character-level transcription accuracy
WER (Word Error Rate)
Measures word-level correctness

Evaluation is performed by comparing model outputs with ground truth transcriptions.

 
 ###Benchmark Comparison
Baseline: Tesseract OCR
Proposed: Transformer OCR + LLM/VLM correction

The proposed pipeline consistently outperforms the baseline across all tested samples.

<img width="1470" height="620" alt="image" src="https://github.com/user-attachments/assets/fda57ed9-3a4c-497e-9ead-7e5bbaea5253" />
