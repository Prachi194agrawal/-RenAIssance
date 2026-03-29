# Historical Spanish Manuscript OCR using Vision-Language Model Fine-Tuning

<p align="center">
  <img src="images/humanai_logo.jpg" alt="HumanAI Foundation" height="80" style="margin-right: 20px;"/>
  &nbsp;&nbsp;&nbsp;&nbsp;
  <img src="images/gsoc_logo.png" alt="Google Summer of Code" height="50"/>
</p>

<p align="center">
  <a href="https://summerofcode.withgoogle.com/programs/2024/projects/lg7vQeMM">
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

> **Developer:** Shashank Shekhar Singh — B.Tech, IIT BHU Varanasi, India  
> **Blog post:** [My Journey with HumanAI in GSoC 2024 — Part 2](https://medium.com/@shashankshekharsingh1205/my-journey-with-humanai-in-the-google-summer-of-code24-program-part-2-bb42abce3495)

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
