# 🗣️ Multilingual Indian TTS:

A fine-tuned **VITS (Variational Inference Text-to-Speech)** model for generating natural-sounding speech in **10 Indian languages**. Built using a Kaggle dataset and customized metadata, this project adapts powerful neural TTS techniques for the rich linguistic diversity of India.

## 🌐 Supported Languages

This model is trained to support the following Indian languages:
- Hindi
- Tamil
- Bengali
- Telugu
- Kannada
- Marathi
- Gujarati
- Punjabi
- Malayalam
- Urdu

## 🚀 Project Highlights

- 🎯 Fine-tuned **VITS**, a state-of-the-art end-to-end TTS model
- 🗃️ Used **Kaggle's "Audio Dataset with 10 Indian Languages"** for training
- 📄 Created a **custom `metadata.csv`** to format training data
- 🧪 Wrote a full training pipeline from scratch using PyTorch

## 🛠️ Tech Stack

- **Python**
- **VITS** (via [ESPnet](https://github.com/espnet/espnet) or [Coqui TTS](https://github.com/coqui-ai/TTS))
- **PyTorch**
- **Librosa** for audio processing
- **NumPy**, **pandas** for metadata handling
- **Kaggle Dataset** for multilingual speech data
