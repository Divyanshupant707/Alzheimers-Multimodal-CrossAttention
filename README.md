# Multimodal Alzheimer’s Disease Prediction via Bidirectional Cross-Attention Networks

This repository contains a deep learning framework designed to predict Alzheimer's Disease (AD) versus Cognitively Normal (CN) states by fusing structural **MRI** and metabolic **PET** neuroimaging data using advanced fusion mechanisms.

## 📂 Repository Structure
* `notebooks/1_MRI_Classification.ipynb`: Baseline classification pipeline using ResNet-18 on structural MRI data.
* `notebooks/2_PET_Classification.ipynb`: Baseline classification pipeline utilizing a Vision Transformer (ViT-Base) on PET scans.
* `notebooks/3_Multimodal_Fusion.ipynb`: Core production notebook implementing a custom Bidirectional Cross-Attention module to join both modalities.

---

## 🧠 Model Architecture
Instead of utilizing simple feature concatenation, this project implements a **Bidirectional Cross-Attention Fusion Network**:
1. **MRI Stream:** Features are extracted using a fully-trainable ResNet-18 backbone.
2. **PET Stream:** Features are processed via a fine-tuned Vision Transformer (`google/vit-base-patch16-224-in21k`).
3. **Cross-Attention Mechanism:** Features interact via `nn.MultiheadAttention`, allowing the structural representations (MRI) to attend to metabolic regions (PET) and vice versa before final classification.

---

## 📊 Dataset Compliance & Pipeline
Data used in this project were obtained from the **Alzheimer's Disease Neuroimaging Initiative (ADNI)** database (adni.loni.usc.edu). 

### Data Leakage Mitigation
To maintain strict clinical validity, a **subject-wise stratified split** was engineered across 100 common subjects. This guarantees that slices belonging to a specific patient are isolated strictly within either the training set ($8,222$ pairs) or the validation set ($1,896$ pairs), preventing target leakage across epochs.

### Training Details
* **Augmentations:** Heavy affine transformations, Gaussian blurring, and color jittering combined with on-the-fly **Mixup data augmentation**.
* **Regularization:** Label smoothing ($0.05$), weight decay ($0.2$), and early stopping monitored on validation loss.
* **Optimization:** Mixed-precision training (`torch.cuda.amp`) executed on an NVIDIA T4 GPU.

---

## 🛠️ How to run
1. Clone the repository:
   ```bash
   git clone [https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git](https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git)
