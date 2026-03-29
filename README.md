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


## Table of Contents

- [Project Goals](#project-goals)
- [Installation](#installation)
- [About The Project](#about-the-project)
- [Datasets and Models](#datasets-and-models)
- [Model Performance](#model-performance)
- [Acknowledgements](#acknowledgements)
- [License](#license)
- [Links](#links)

## Project Goals

1. **Fine-Tuning a Vision-Language Model for Historical OCR:** The primary goal is to adapt `Qwen2.5-VL-7B-Instruct`, a large multimodal model, for the specialized task of transcribing 17th-century Spanish manuscripts. Using parameter-efficient LoRA fine-tuning via the Unsloth framework, the model is trained to capture both visual glyph-level features and the contextual orthographic patterns unique to Early Modern Spanish printing.

2. **Achieving Superior Accuracy over Baseline OCR:** The objective is to significantly outperform conventional OCR tools like Tesseract on historical Spanish documents. This involves paleographic normalization rules, data augmentation, and rigorous hyperparameter tuning to ensure the model generalises across diverse typefaces, degradation levels, and scribal conventions present in corpus documents dating from 1628 to 1650.

## Installation

You don't need to install anything externally. Fire up the Python notebooks on your favourite coding platform (Google Colab, Jupyter Notebook, Kaggle, etc.) and run cells sequentially. All required packages are installed by the first code cell in each notebook.

### Project Directory Structure

1. **step_1_model_training.ipynb** — End-to-end pipeline for data ingestion, pre-processing, augmentation, and LoRA fine-tuning of Qwen2.5-VL-7B. Run this first to produce the trained adapter (`My_Best_Qwen_LoRA/`).

2. **step_2_model_checking.ipynb** — Standalone evaluation notebook. Loads the saved LoRA adapter, runs inference on the validation split, computes CER/WER, compares against a Tesseract baseline, and launches an interactive Gradio UI for live transcription.

### Key Dependencies

```bash
# Installed automatically by the notebooks
pip install "trl>=0.18.2,<=0.24.0"
pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
pip install --upgrade transformers accelerate peft
pip install pymupdf python-docx jiwer rapidfuzz nltk datasets
pip install qwen-vl-utils gradio pyspellchecker
apt-get install poppler-utils tesseract-ocr tesseract-ocr-spa
```

## About The Project

#### Paleographic Irregularities and Transcription Rules

The model is trained and prompted with the following corpus-specific conventions:

- **Interchangeable Characters**: `u`/`v` and `f`/long-`s` (ſ) were graphically ambiguous in 17th-century type. Transcription preserves the visible glyph without modernisation.
- **Macron / Tilde Abbreviations**:
    1. A macron or tilde over a vowel expands to `+n` (e.g., `ã → an`, `ẽ → en`)
    2. A tilde over `q` expands to `que` (q̃ → que)
    3. `ñ` is always preserved exactly — the tilde over `n` is semantically meaningful.
- **Old Spellings**: `ç` always maps to modern `z` (Ç → Z, ç → z).
- **Line-End Hyphens**: Split words at line boundaries are transcribed as written and not rejoined.
- **Accent Marks**: Transcribed exactly as visible; accents are inconsistent in the corpus and are not normalised.

#### Dataset and Pre-processing

- **Input Corpus:** Six scanned PDF source documents paired with six DOCX ground-truth transcriptions, covering historical Spanish texts from 1628–1650:
    - *Buendia - Instruccion*
    - *Covarrubias - Tesoro lengua*
    - *Guardiola - Tratado nobleza*
    - *PORCONES.228.38 – 1646*
    - *PORCONES.23.5 - 1628*
    - *PORCONES.748.6 – 1650*

- **PDF-to-Image Extraction**: Pages are rendered at 150 DPI using PyMuPDF, resized to a maximum of 768×1024 px, and saved as JPEG. Contrast enhancement (factor 1.4) and sharpening are applied to improve legibility.
    ```
    Total pages extracted: 60  (10 pages per document)
    ```

- **Transcription Parsing**: DOCX transcription files are parsed by detecting page markers of the form `PDFp{N}`, splitting each document into per-page ground-truth text blocks. Fuzzy name matching (SequenceMatcher, cutoff 0.55) automatically pairs PDFs with their corresponding DOCX.

- **Data Augmentation**: Each original image is augmented to produce 3 additional training variants using:
    - Random rotation (±10°)
    - Random crop (85–97% of original area, resized back)
    - Contrast enhancement + sharpening
    ```
    Final dataset: 76 paired samples (original + augmented)
    Train / Val split: 85% / 15%  (seed = 42)
    Train: 65 samples  |  Val: 11 samples
    ```

#### Model Architecture

- **Base Model**: `Qwen2.5-VL-7B-Instruct` — a 7-billion parameter vision-language model quantised to 4-bit precision (`bnb-4bit`) via the Unsloth framework for memory-efficient training on a single T4 GPU (15.6 GB VRAM).

- **Fine-Tuning Strategy — LoRA**: Parameter-efficient Low-Rank Adaptation is applied across all core projection layers. Vision layers, language layers, attention modules, and MLP modules are all fine-tuned simultaneously.

    | LoRA Hyperparameter | Value |
    |---|---|
    | Rank (`r`) | 16 |
    | Alpha (`lora_alpha`) | 32 |
    | Dropout | 0.05 |
    | Target modules | `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj` |
    | RSLoRA | Disabled |

- **System Prompt**: The model is conditioned at inference time with a detailed 7-rule paleographic prompt covering exact-copy transcription, u/v and f/ſ disambiguation, abbreviation expansion, ç→z substitution, and line-end hyphen handling.

- **Post-Processing Normalization**: A deterministic normalization pipeline runs on model output to expand macron/tilde abbreviations, handle precomposed Unicode characters, and apply the u/v disambiguation heuristic via a Spanish spell-checker.

    ```
    Normalization self-tests (all passed):
      'mãdato del rey'  →  'mandato del rey'
      'q̃ se haga'       →  'que se haga'
      'la çiudad'       →  'la ziudad'
      'el señor'        →  'el señor'
    ```

#### Training and Evaluation

- **Training Configuration**:

    | Hyperparameter | Value |
    |---|---|
    | Epochs | 8 |
    | Batch size (train/eval) | 1 |
    | Gradient accumulation steps | 4 |
    | Learning rate | 2e-4 |
    | LR scheduler | Cosine |
    | Optimizer | AdamW 8-bit (GPU) |
    | Weight decay | 0.01 |
    | Warmup steps | 5 |
    | Max sequence length | 2048 |
    | Precision | fp16 (T4) / bf16 (Ampere+) |
    | Best model selection | Minimum `eval_loss` |

- **Conversation Format**: Each training sample is formatted as a three-turn conversation — a system message with paleographic rules, a user message containing the manuscript image and transcription instruction, and an assistant response with the ground-truth transcription.

- **Evaluation Metrics**: Performance is evaluated using Character Error Rate (CER) and Word Error Rate (WER) via the `jiwer` library, computed on the held-out 15% validation split (11 samples). The fine-tuned VLM is benchmarked against Tesseract 4.1.1 (`--oem 1 --psm 6 -l spa`) as a classical OCR baseline.

- **Gradio Demo**: An interactive UI is included in the evaluation notebook. Upload any manuscript image to receive a live transcription with an annotated output panel.

<!-- TODO: Add screenshot of Gradio UI if available -->

## Datasets and Models

- Source PDFs and DOCX transcriptions are stored in Google Drive under `MyDrive/print_pdf/` and `MyDrive/Print/` respectively. See the training notebook for exact path configuration.
- The trained LoRA adapter is saved automatically to `MyDrive/Saved_PDF_Data/My_Best_Qwen_LoRA/` and is loaded by the evaluation notebook without retraining.
- Evaluation results (per-sample CER/WER, predictions vs. references) are exported to `MyDrive/Saved_PDF_Data/Evaluation_Results/` as both JSON and CSV.

## Model Performance

The fine-tuned Qwen2.5-VL-7B + LoRA model substantially outperforms the Tesseract OCR baseline on the 11-sample validation set:

| Metric | Tesseract (Baseline) | Qwen2.5-VL + LoRA (Ours) | Improvement |
|--------|---------------------|--------------------------|-------------|
| Overall CER | 0.628 (62.8%) | 0.485 (48.5%) | **↓ 22.8%** |
| Overall WER | 1.003 (100.3%) | 0.627 (62.7%) | **↓ 37.5%** |

**Per-sample highlights:**

| Image | Tess CER | VLM CER | Δ CER | Tess WER | VLM WER | Δ WER |
|---|---|---|---|---|---|---|
| page0001_aug2_rot_contrast | 1.607 | 0.506 | ↓ 1.101 | 2.855 | 0.711 | ↓ 2.145 |
| PORCONES.228.38_page0004 | 0.205 | 0.059 | ↓ 0.146 | 0.667 | 0.145 | ↓ 0.522 |
| page0001_aug1_rot_contrast | 1.653 | 0.010 | ↓ 1.643 | 2.513 | 0.066 | ↓ 2.447 |
| Buendia_-_Instruccion_page0004 | 0.305 | 0.145 | ↓ 0.159 | 0.800 | 0.285 | ↓ 0.515 |

> **Note:** The VLM underperforms Tesseract on a small number of heavily augmented samples (e.g., severely rotated + cropped pages), where degradation exceeds the training distribution. Overall, the model provides meaningful improvement across the corpus.

<!-- TODO: Add bar chart image (evaluation_comparison.png) from Evaluation_Results/ folder if exporting to GitHub -->
Quick Start (Run Directly)
 Training Notebook
 
 https://colab.research.google.com/drive/1GAVAzlilZEp4w0cvwrviRCPhHwOvWsY9?usp=sharing

 Evaluation Notebook

 
https://colab.research.google.com/drive/1LuC_b0QEr5DirhDEraxZQhCoUM8VchyR?usp=sharing


 Dataset Access

 https://drive.google.com/drive/folders/1S1eC18EHBkXIJ4AJmLdT-fOzwxWvEWNC?usp=sharing

## Acknowledgements

This project is supported by the [HumanAI Foundation](https://humanai.foundation/) and Google Summer of Code 2024. A detailed walkthrough of the project's development, challenges, and solutions can be found on the [blog post](https://medium.com/@shashankshekharsingh1205/my-journey-with-humanai-in-the-google-summer-of-code24-program-part-2-bb42abce3495).

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Links

- [Google Summer of Code 2024 Project](https://summerofcode.withgoogle.com/programs/2024/projects/lg7vQeMM)
- [HumanAI Foundation](https://humanai.foundation/)

Feel free to fork the repository and submit pull requests. For major changes, please open an issue first to discuss your ideas. Contributions are always welcome!
