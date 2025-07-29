# T5 Paraphrase Generation

This project demonstrates how to fine-tune a T5 (Text-to-Text Transfer Transformer) model for paraphrase generation using Hugging Face Transformers.

## 🚀 Overview

The goal is to train a model that can take an input sentence and generate a semantically equivalent paraphrased sentence.

## 🧰 Requirements

- Python >= 3.7
- PyTorch
- Hugging Face Transformers
- Datasets
- SentencePiece

Install requirements with:

```bash
pip install transformers datasets sentencepiece




 Project Structure

t5_paraphraser/
│
├── train.py            # Main training script
├── inference.py        # Run paraphrase generation
├── paraphrase_dataset.csv  # (optional) Your custom dataset
├── README.md           # Project documentation
└── logs/               # Training logs

