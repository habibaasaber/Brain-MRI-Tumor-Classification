# Brain MRI Tumor Detection
**Deep Learning Project - CAI3105 | AASTMT Smart Village**

## 📋 Project Overview

Comparative study evaluating two deep learning paradigms for brain MRI tumor classification:
- **Approach 1**: DL-based Feature Learning + SVM Classifier
- **Approach 2**: End-to-End Fine-tuned Deep Learning

Using **EfficientNetB0** and **ResNet50** (bonus) as pretrained backbones on a 7,200-image dataset with 4 tumor classes: Glioma, Meningioma, No Tumor, and Pituitary.

---

## 🎯 Key Results

| Architecture | Approach | Accuracy | Training Time | Parameters |
|-------------|----------|----------|---------------|------------|
| **EfficientNetB0 + SVM** | Feature Extraction | **94.2%** | 44.40 sec | 5.3M |
| **EfficientNetB0 E2E** | Fine-tuning | **94.4%** | 159.02 sec | 5.3M |
| ResNet50 + SVM | Feature Extraction | 92.0% | 69.22 sec | 25.6M |
| ResNet50 E2E | Fine-tuning | 93.0% | 237.24 sec | 25.6M |

**Winner**: EfficientNetB0 achieves superior accuracy-to-efficiency trade-off (3.6× faster than ResNet50 with 1.4% higher accuracy)

---

## 📁 Dataset

**Source**: [Brain MRI Images for Brain Tumor Detection](https://www.kaggle.com/navoneel/brain-mri-images-for-brain-tumor-detection) (Kaggle)

- **Total Samples**: 7,200 MRI scans
- **Training Set**: 5,600 images (77.8%)
- **Testing Set**: 1,600 images (22.2%)
- **Classes**: 4 (Glioma, Meningioma, No Tumor, Pituitary)
- **Image Size**: 224 × 224 × 3 (RGB)

---

## 🏗️ Architecture

### Approach 1: DL Feature Extraction + SVM
