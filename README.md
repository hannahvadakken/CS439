# CS439
Final Project

# PEFT-Based Industrial Acoustic Fault Detection

This project explores the use of Parameter-Efficient Fine-Tuning (PEFT) techniques for industrial acoustic anomaly detection using transformer-based audio models. Specifically, the project adapts the Audio Spectrogram Transformer (AST) with Low-Rank Adaptation (LoRA) to identify abnormal fan behavior within the MIMII dataset.

The goal of the project is not to create a production-ready predictive maintenance system, but rather to investigate whether lightweight transformer fine-tuning can be applied to industrial monitoring tasks using limited hardware and relatively small datasets.

## Overview

Industrial downtime remains a major financial issue across manufacturing and production environments. Detecting abnormal machine behavior before catastrophic failure can help reduce maintenance costs and prevent unexpected outages.

This project uses:

- Audio Spectrogram Transformer (AST)
- Low-Rank Adaptation (LoRA)
- Masked autoencoder reconstruction
- Log-mel spectrogram preprocessing
- The MIMII industrial sound dataset

The model was trained on normal fan recordings and evaluated on both normal and abnormal machine audio samples.

## Results

The final model achieved approximately **67\% classification accuracy** on the evaluation dataset. While the model demonstrated some ability to distinguish between normal and abnormal acoustic patterns, the results also highlight the difficulty of industrial anomaly detection under constrained data and hardware conditions.

Rather than presenting state-of-the-art performance, this project serves as an exploratory implementation of PEFT-based transformer adaptation for industrial audio monitoring.

## Features

- Transformer-based acoustic anomaly detection
- Lightweight PEFT fine-tuning with LoRA
- Log-mel spectrogram preprocessing pipeline
- Masked autoencoder training framework
- Compatible with consumer-grade hardware
- Focus on reconstruction error rather than explicit fault labels

## Dataset

This project uses the MIMII dataset:

> Purohit et al., 2019 — Malfunctioning Industrial Machine Investigation and Inspection Dataset

The experiments in this repository focus specifically on the fan subset of the dataset.

## Installation

Clone the repository:

```bash
git clone <your-repository-url>
cd <repository-name>
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Please refer to `requirements.txt` for the complete list of required Python packages and library versions.

## Project Structure

```text
project/
│
├── requirements.txt       # Python dependencies
├── README.md
└── main.ipynb             #All the python, including preprocessing scripts, model training, and evaluation
```
## License

This project is intended for academic and research purposes.
