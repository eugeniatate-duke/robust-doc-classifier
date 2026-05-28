# Robust Document Classifier

A computer vision prototype for robust document image classification using transfer learning and data augmentation.

This project was developed for Mini Hackathon #1 in AIPI 540: Deep Learning Applications at Duke University.

## Problem Statement

Business documents are often uploaded under imperfect real-world conditions. Images may be blurry, rotated, shadowed, poorly cropped, distorted, or captured from mobile devices at unusual angles.

The goal of this project is to build a document classification system that can better generalize to noisy and imperfect document images using:

- Transfer learning
- Domain-specific data augmentation
- Robustness-focused evaluation

Rather than maximizing raw accuracy, this project focuses on improving model robustness under real-world variation.

---

# Project Overview

This prototype classifies document images into several document categories:

- Letter
- Form
- Email
- Handwritten
- Advertisement
- Invoice

The system uses a pretrained MobileNetV2 convolutional neural network and fine-tunes a custom classifier head for document classification.

---

# Transfer Learning Approach

The project uses MobileNetV2 pretrained on ImageNet as the feature extractor.

Transfer learning steps:

1. Load pretrained MobileNetV2 weights
2. Freeze pretrained backbone layers
3. Replace final classification layer
4. Train custom classifier head on document images

This approach allows effective training with limited data and reduced training time.

---

# Data Augmentation Strategy

The project applies realistic augmentations designed to simulate common document upload issues:

- Random rotation
- Perspective distortion
- Random resized crop
- Brightness and contrast jitter
- Gaussian blur

These augmentations were selected to improve robustness to:

- Mobile phone captures
- Poor scans
- Cropped documents
- Lighting variation
- Low-quality images

---

# Dataset

Dataset used:
- RVL-CDIP Mini Dataset (subset) https://adamharley.com/rvl-cdip/ 

The project uses a reduced subset of the RVL-CDIP document image dataset for rapid experimentation during the hackathon.

---

# Model Architecture

- Backbone: MobileNetV2
- Framework: PyTorch
- Input size: 224x224
- Loss function: CrossEntropyLoss
- Optimizer: Adam

---

# Results

Two models were trained:

1. Baseline model
2. Augmented model

The augmented model demonstrated improved robustness to distorted and noisy document images.

Some visually similar classes such as emails and invoices remained challenging due to similar layouts and text-heavy structure.

This highlights a key real-world challenge in document computer vision systems.

---

# Project Structure

```text
robust-doc-classifier/
├── app.py
├── train.py
├── evaluate.py
├── requirements.txt
├── README.md
├── src/
├── models/
└── figures/
```

---

# Running the Project

## Install dependencies

```bash
pip install -r requirements.txt
```

## Train baseline model

```bash
python train.py --epochs 3 --max-per-class 50
```

## Train augmented model

```bash
python train.py --augmented --epochs 3 --max-per-class 50
```

## Evaluate model

```bash
python evaluate.py --model-path models/augmented_model.pt --name augmented
```

## Run Gradio app locally

```bash
python app.py
```

---

# Deployment

The project is deployed using Hugging Face Spaces with Gradio.

Live Demo:
- https://huggingface.co/spaces/Eugenia-Tate-Duke/robust-doc-classifier 

---

# Future Improvements

Potential future improvements include:

- Fine-tuning the full backbone
- Using larger document datasets
- Adding Optical Character Recognition features
- Using transformer-based vision models
- Improving visually similar class separation
- Expanding robustness evaluation

---

# Technologies Used

- Python
- PyTorch
- Torchvision
- Hugging Face Datasets
- Gradio
- Scikit-learn
- Matplotlib

---

# Author

Eugenia Tate

AIPI 540 — Deep Learning Applications  
Duke University