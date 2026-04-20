# Pneumonia Detection using SVM

## Overview
This project focuses on classifying chest X-ray images to detect pneumonia using Support Vector Machine (SVM).

Two approaches are implemented:
- A custom Soft-Margin SVM built from scratch using NumPy
- A library-based SVM using Scikit-learn

## Dataset
- Chest X-Ray Images (Pneumonia)
- Source: Kaggle [Chest X-Ray Images (Pneumonia)](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia)
- Images resized to **128x128**
- Classes:
  - NORMAL
  - PNEUMONIA

## Methods

### 1. Custom SVM (From Scratch)
- Implemented using NumPy
- Uses **Soft Margin SVM**
- Optimized with **Stochastic Gradient Descent (SGD)**
- Includes regularization

### 2. Sklearn SVM
- Model: `SVC`
- Kernel: Linear
- Parameter: `C = 0.01`

## Evaluation Metrics
- Precision
- Recall
- F1-Score

## Results Summary

| Model         | Precision | Recall | F1-score |
|--------------|----------|--------|----------|
| Custom SVM   | Higher   | High   | Higher   |
| Sklearn SVM  | Lower    | High   | Lower    |

## Observations
- Custom SVM performs better in **Precision** and **F1-score**
- Both models achieve high **Recall**
- Sklearn SVM is significantly faster due to optimized implementation (C++ & SMO algorithm)

## Author
Nguyễn Minh Triết
