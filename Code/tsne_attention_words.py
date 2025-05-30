import torch
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from nltk.tokenize import TreebankWordTokenizer
import numpy as np
import os

from attention_mapping import generate_caption_with_attention
from vitcap_model import PatchAttentionDecoder
from utils import load_vocab
from PIL import Image
from torchvision import transforms

# ------------------
# Load model & vocab
# ------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
vocab = load_vocab("/home/pdas4/vitcap_r/data/coco_vocab.pkl")
idx2word = {i: w for w, i in vocab.items()}

model = PatchAttentionDecoder(embed_dim=256, hidden_dim=256, vocab_size=len(vocab))
checkpoint = torch.load("/home/pdas4/vitcap_r/models/patch_attention_best.pth", map_location=device)
filtered_ckpt = {k: v for k, v in checkpoint.items() if not k.startswith("vit.")}
model.load_state_dict(filtered_ckpt, strict=False)
model.to(device)

# ------------------
# Image + transform
# ------------------
image_path = "/home/pdas4/vitcap_r/data/MSCOCO/val2014/COCO_val2014_000000000488.jpg"
image = Image.open(image_path).convert("RGB")
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5]*3, std=[0.5]*3)
])
image_tensor = transform(image).to(device)

# ------------------
# Get attention maps
# ------------------
tokens, attn_weights = generate_caption_with_attention(model, image_tensor, vocab, device)
words = [idx2word[t] for t in tokens if t not in {vocab['startseq'], vocab['endseq'], vocab['<pad>']}]

# Remove CLS tokens and flatten
attn_vectors = [attn[1:].reshape(-1).detach().cpu().numpy() for attn in attn_weights[:len(words)]]

# ------------------
# Run t-SNE
# ------------------
tsne = TSNE(n_components=2, perplexity=5, random_state=42)
attn_matrix = np.stack(attn_vectors) 
embeds_2d = tsne.fit_transform(attn_matrix)

# ------------------
# Plot
# ------------------
plt.figure(figsize=(10, 7))
for i, word in enumerate(words):
    x, y = embeds_2d[i]
    plt.scatter(x, y, color='blue')
    plt.text(x + 0.5, y + 0.5, word, fontsize=9)

plt.title("🧬 t-SNE of Word Attention Maps")
plt.axis('off')
plt.tight_layout()
plt.savefig("attention_maps/tsne_attention_words.png", dpi=300)
print("✅ Saved: attention_maps/tsne_attention_words.png")
