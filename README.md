# ViTCap-R: Vision Transformer-Based Image Captioning with Dual-Encoder Retrieval

ViTCap-R is a unified deep learning framework that combines image captioning and image-text retrieval. It integrates Vision Transformers, patch-level attention, and dual-encoder contrastive learning.

## 🚀 Features
- Vision Transformer (ViT) encoder
- Patch-level attention decoder (LSTM-based)
- Dual-encoder retrieval with contrastive InfoNCE loss
- Hard Negative Mining (HNM) support
- Qualitative visualizations (attention heatmaps, t-SNE, PCA)
- Evaluation with BLEU, METEOR, and Recall@K

## 📦 Implementation Details
See the Project Report for implementation details, including system architecture, training methodology, dataset information, experimental setup, and evaluation results.

## 📊 Datasets
- [Flickr8k](https://forms.illinois.edu/sec/1713398)
- [MS COCO 2014](https://cocodataset.org/#download)

## 🧠 Models
Pretrained models are available in the `models/` directory:
- `patch_attention_best.pth`
- `dual_encoder_best.pth`
- `dual_encoder_hnm_best.pth`
- `model_weights_vit25.h5`

## 🛠️ Setup
```bash
git clone https://github.com/Prayash-Das/ViTCap-R.git
cd ViTCap-R
pip install -r requirements.txt
