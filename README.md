# AI for Healthcare: Smarter Tools Against Skin Cancer 🩺🔬

## Overview

Skin cancer is one of the most common and rapidly increasing forms of cancer worldwide. Early detection plays a crucial role in improving treatment outcomes and survival rates.

This project presents an AI-powered skin cancer detection system that uses advanced deep learning techniques to classify skin lesions as **Benign** or **Malignant** from dermoscopic images. The system combines modern architectures such as **ResNet**, **EfficientNet**, and **Vision Transformer (ViT)** to achieve accurate predictions and provide a user-friendly screening solution.

The trained model is integrated into a web-based application where users can upload skin lesion images and receive real-time predictions along with confidence scores.

---

## Key Features ✨

- Skin lesion classification (Benign vs Malignant)
- Deep Learning based prediction system
- ResNet Architecture
- EfficientNet Architecture
- Vision Transformer (ViT)
- Real-time image analysis
- User-friendly web interface
- Confidence score generation
- Explainable AI using Grad-CAM
- Medical image preprocessing pipeline
- Data augmentation techniques
- Transfer learning implementation

---

## Problem Statement

Traditional skin cancer diagnosis often requires specialized medical equipment and experienced dermatologists. Access to such facilities is limited in many regions, resulting in delayed diagnosis.

This project aims to provide an accessible AI-based preliminary screening tool that can assist in the early detection of skin cancer and encourage timely medical consultation.

---

## Technologies Used 🛠️

### Programming Language
- Python

### Deep Learning Frameworks
- TensorFlow
- Keras

### Web Development
- Flask
- HTML
- CSS
- Bootstrap

### Data Processing
- NumPy
- Pandas
- Pillow

### Visualization
- Matplotlib
- Grad-CAM

### Development Environment
- Visual Studio Code

---

## Dataset 📊

The project utilizes the **HAM10000 Dataset**, a publicly available collection of dermoscopic skin lesion images.

### Data Preparation

- Image Resizing
- Normalization
- Data Augmentation
  - Rotation
  - Flipping
  - Zooming
  - Shifting

These preprocessing steps improve model generalization and reduce overfitting.

---

## System Architecture

User
  │
  ▼
Web Interface (Flask)
  │
  ▼
Image Upload
  │
  ▼
Preprocessing
  │
  ▼
Deep Learning Model
(ResNet / EfficientNet / ViT)
  │
  ▼
Prediction
  │
  ▼
Confidence Score
  │
  ▼
Result Display

---

## Workflow

1. User uploads a skin lesion image.
2. Image is validated.
3. Preprocessing is performed.
4. Image is resized and normalized.
5. Deep learning model analyzes the image.
6. Features are extracted.
7. Classification is performed.
8. Prediction and confidence score are generated.
9. Results are displayed to the user.

---

## Deep Learning Models

### ResNet
- Residual learning architecture
- Handles deep networks efficiently
- Strong feature extraction capability

### EfficientNet
- Compound scaling approach
- High accuracy with fewer parameters
- Computationally efficient

### Vision Transformer (ViT)
- Attention-based architecture
- Captures global image relationships
- Effective for medical image classification

---

## Explainable AI (XAI)

To improve transparency and trust, the project incorporates **Grad-CAM (Gradient-weighted Class Activation Mapping)**.

Grad-CAM highlights important regions of the image that influenced the model’s decision, helping users understand the prediction process.

---

## Installation

### Clone Repository

git clone https://github.com/apeksha124/-LesNet-main.git
cd -LesNet-main

### Create Virtual Environment

python -m venv venv

### Activate Environment

Windows:

venv\Scripts\activate

### Install Dependencies

pip install -r requirements.txt

---

## Running the Project

python app.py

or

python app_production_fixed.py

Open:

http://127.0.0.1:5000

in your browser.

---

## Performance Highlights

- High classification accuracy
- Real-time prediction capability
- Confidence score generation
- Explainable AI support
- Lightweight deployment through Flask

---

## Project Structure

LesNet-main
│
├── Dataset/
├── data/
├── models/
├── tests/
├── app_production_fixed.py
├── ml_logic.py
├── ui_components.py
├── utils.py
├── requirements.txt
├── README.md
└── .gitignore

---

## Future Enhancements 🚀

- Multi-class skin disease classification
- Mobile application development
- Cloud deployment
- Doctor consultation integration
- Larger dataset training
- Improved explainability techniques
- Real-time camera detection

---

## Educational Purpose

This project is intended as an AI-assisted screening tool and should not be considered a replacement for professional medical diagnosis.

Users are encouraged to consult qualified healthcare professionals for medical advice and confirmation.

---

## Authors 👩‍💻
- Apeksha Tiwari
---
