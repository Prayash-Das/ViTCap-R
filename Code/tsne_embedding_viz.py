import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from tqdm import tqdm
import os

os.makedirs("/home/pdas4/vitcap_r/figures", exist_ok=True)
# ---------------------------
# Load Features (Edit paths)
# ---------------------------
img_embed_path = "/home/pdas4/vitcap_r/embeddings/img_embeds.pt"
txt_embed_path = "/home/pdas4/vitcap_r/embeddings/txt_embeds.pt"
hard_neg_path = "/home/pdas4/vitcap_r/embeddings/hardest_neg_indices.pt"  # Optional

img_embeds = torch.load(img_embed_path)
txt_embeds = torch.load(txt_embed_path)
hard_neg_indices = torch.load(hard_neg_path) if hard_neg_path else None

# Convert to numpy
img_np = img_embeds.cpu().numpy()
txt_np = txt_embeds.cpu().numpy()

# Stack and apply t-SNE
print("🔍 Running t-SNE projection...")
all_vecs = np.vstack([img_np, txt_np])
tsne = TSNE(n_components=2, random_state=42, perplexity=30)
vecs_2d = tsne.fit_transform(all_vecs)

# Split back
img_2d = vecs_2d[:len(img_np)]
txt_2d = vecs_2d[len(img_np):]

# ---------------------------
# Plot
# ---------------------------
plt.figure(figsize=(12, 10))

# Scatter points
plt.scatter(img_2d[:, 0], img_2d[:, 1], c='blue', label='Images', alpha=0.6, s=10)
plt.scatter(txt_2d[:, 0], txt_2d[:, 1], c='red', label='Captions', alpha=0.6, s=10)

# Matching lines
N=300
'''for i in range(len(img_2d)):
    plt.plot([img_2d[i, 0], txt_2d[i, 0]],
             [img_2d[i, 1], txt_2d[i, 1]],
             c='gray', linewidth=0.4, alpha=0.3)

# Hard negative lines (optional)
if hard_neg_indices is not None:
    for i in range(len(img_2d)):
        j = hard_neg_indices[i]
        plt.plot([img_2d[i, 0], txt_2d[j, 0]],
                 [img_2d[i, 1], txt_2d[j, 1]],
                 c='black', linestyle='--', linewidth=0.7, alpha=0.4)'''

# Matching lines: image ↔ caption
for i in range(min(N, len(img_2d))):
    plt.plot([img_2d[i, 0], txt_2d[i, 0]],
             [img_2d[i, 1], txt_2d[i, 1]],
             c='gray', linewidth=0.4, alpha=0.4)

# Hard negative lines: image ↔ confusing caption
if hard_neg_indices is not None:
    for i in range(min(N, len(img_2d))):
        j = hard_neg_indices[i]
        plt.plot([img_2d[i, 0], txt_2d[j, 0]],
                 [img_2d[i, 1], txt_2d[j, 1]],
                 c='black', linestyle='--', linewidth=0.7, alpha=0.4)
plt.title("t-SNE Projection: Dual Encoder Image-Text Embedding Space")
plt.legend(loc='upper right', fontsize='small')
plt.grid(True)
plt.tight_layout()

# Save to file
save_path = "/home/pdas4/vitcap_r/figures/tsne_vitcap_retrieval.png"
os.makedirs(os.path.dirname(save_path), exist_ok=True)
plt.savefig(save_path, dpi=300)
print(f"✅ Plot saved to: {save_path}")
plt.show()
