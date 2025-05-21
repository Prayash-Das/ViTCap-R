# ViTCap-R: Vision Transformer-Based Image Captioning with Dual-Encoder Retrieval

ViTCap-R is a unified deep learning framework that combines image captioning and image-text retrieval. It integrates Vision Transformers, patch-level attention, and dual-encoder contrastive learning.

## 🚀 Features
- Vision Transformer (ViT) encoder
- Patch-level attention decoder (LSTM-based)
- Dual-encoder retrieval with contrastive InfoNCE loss
- Hard Negative Mining (HNM) support
- Qualitative visualizations (attention heatmaps, t-SNE, PCA)
- Evaluation with BLEU, METEOR, and Recall@K

## 📦 Project Structure
See directory layout in the repo.

## 📊 Datasets
- [Flickr8k](https://forms.illinois.edu/sec/1713398)
- [MS COCO 2014](https://cocodataset.org/#download)

## 🧠 Models
Pretrained models are available in the `models/` directory:
- `patch_attention_best.pth`
- `dual_encoder_best.pth`

## 🛠️ Setup
```bash
git clone https://github.com/<your_username>/ViTCap-R.git
cd ViTCap-R
pip install -r requirements.txt
