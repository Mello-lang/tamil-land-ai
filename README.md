# Tamil Land AI & HWCR

This repository contains tools and machine learning pipelines for **Tamil Land Record Information Extraction (NER)** and **Tamil Handwritten Character Recognition (HWCR)**.

## Project Structure

- **`tamil-land-ai/`**: Tamil Land Record Named Entity Recognition (NER) pipeline using transformer models (IndicBERT).
  - `nlp/generate_dataset_v3.py`: Synthetic dataset generator for realistic Tamil land records (patta, survey numbers, owner names, land extent, boundaries).
  - `nlp/train_ner_v2.py`: Training script with evaluation and metric tracking.
  - `nlp/predict_v3.py`: Inference pipeline for extracting entities from Tamil text.
  - `data/`: Dataset splits and entity label mappings.
- **`tamil-hwcr/`**: Tamil Handwritten Character Recognition dataset annotations and evaluation scripts.

## Setup & Requirements

```bash
# Setup virtual environment
python -m venv .venv
source .venv/bin/activate  # Or on Windows: .venv\Scripts\activate

# Install dependencies
pip install torch transformers datasets scikit-learn seqeval
```

## Running Land NER Dataset Generation & Training

```bash
# Generate Dataset
python tamil-land-ai/nlp/generate_dataset_v3.py

# Train NER Model
python tamil-land-ai/nlp/train_ner_v2.py

# Run Prediction / Inference
python tamil-land-ai/nlp/predict_v3.py
```
