import torch
import matplotlib.pyplot as plt
import torch.nn.functional as F

# Load embeddings
img_tensor = torch.load("/home/pdas4/vitcap_r/embeddings/img_embeds.pt")
txt_tensor = torch.load("/home/pdas4/vitcap_r/embeddings/txt_embeds.pt")
hardest_neg_indices = torch.load("/home/pdas4/vitcap_r/embeddings/hardest_neg_indices.pt")

# Ensure same device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
img_tensor = img_tensor.to(device)
txt_tensor = txt_tensor.to(device)
hardest_neg_indices = hardest_neg_indices.to(device)

# 🔹 Positive pair similarities
pos_scores = F.cosine_similarity(img_tensor, txt_tensor)

# 🔸 Hard negative similarities
hard_neg_txt = txt_tensor[hardest_neg_indices]
neg_scores = F.cosine_similarity(img_tensor, hard_neg_txt)

# 📊 Plot histogram
plt.figure(figsize=(10, 6))
plt.hist(pos_scores.cpu().numpy(), bins=50, alpha=0.6, label="Positive Pairs", color='green')
plt.hist(neg_scores.cpu().numpy(), bins=50, alpha=0.6, label="Hard Negatives", color='red')
plt.axvline(pos_scores.mean().item(), color='green', linestyle='dashed', linewidth=1)
plt.axvline(neg_scores.mean().item(), color='red', linestyle='dashed', linewidth=1)

plt.title("Cosine Similarity: Positive vs. Hard Negatives")
plt.xlabel("Cosine Similarity")
plt.ylabel("Frequency")
plt.legend()
plt.grid(True)

# Save figure
plt.tight_layout()
plt.savefig("/home/pdas4/vitcap_r/figures/embedding_distance_histogram.png", dpi=300)
print("✅ Saved to /home/pdas4/vitcap_r/figures/embedding_distance_histogram.png")
plt.show()
