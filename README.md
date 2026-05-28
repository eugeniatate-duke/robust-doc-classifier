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
2. Freeze most pretrained backbone layers
3. Unfreeze the final MobileNetV2 blocks for domain adaptation
4. Replace the final classification layer
5. Fine-tune the classifier head on document images

This approach allowed the pretrained model to better adapt from natural images to document-layout classification while still benefiting from ImageNet feature learning.
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
- Transfer learning with partial backbone fine-tuning

---

# Results

Two transfer learning models were evaluated:

1. Baseline model: 76.2% accuracy
2. Augmented model: 67.5% accuracy

* Transfer learning significantly improved document classification performance.
* The baseline model achieved the strongest overall performance.
* The augmented model demonstrated some robustness improvements but aggressive augmentations reduced performance for several visually subtle document classes.
* Visually similar classes such as invoices, forms, and letters remained challenging due to overlapping layout structures.

Per-Class Observations

Strongest classes:

* Email
* Handwritten
* Advertisement

Most difficult classes:

* Form
* Invoice

The baseline model achieved:

* 94.7% recall for emails
* 100% recall for handwritten documents
* 68% recall for invoices

One important observation from this project is that pure CNN-based document classifiers rely heavily on layout and visual structure rather than semantic text understanding.

Although the word “INVOICE” may visibly appear in a document image, the model does not explicitly read text because OCR was not used in this prototype.

This explains why visually similar business documents can still be confused despite strong overall performance.

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
python train.py --epochs 10 --max-per-class 150
```

## Train augmented model

```bash
python train.py --augmented --epochs 10 --max-per-class 150
```

## Evaluate model

```bash
python evaluate.py --model-path models/baseline_model.pt --name baseline
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

* OCR integration for text-aware classification
* Vision Transformer (ViT) architectures
* Multimodal document understanding
* Larger and more balanced datasets
* More targeted augmentation strategies
* Fine-tuning additional backbone layers

---

# Technologies Used

- Python
- PyTorch
- Torchvision
- Hugging Face Datasets
- Gradio
- Scikit-learn
- Matplotlib
- MobileNetV2 Transfer Learning

---

# Author

Eugenia Tate

AIPI 540 — Deep Learning Applications  
Duke University