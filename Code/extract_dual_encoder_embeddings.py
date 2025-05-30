import torch
from tqdm import tqdm
from utils import get_loader, load_vocab
from dual_encoder_model_hnm import DualEncoder
import torch.nn.functional as F
import os

# ---------------------
# Config
# ---------------------
model_path = "/home/pdas4/vitcap_r/models/dual_encoder_hnm_best.pth"
vocab_path = "/home/pdas4/vitcap_r/data/coco_vocab.pkl"
image_dir = "/home/pdas4/vitcap_r/data/MSCOCO/val2014"
annotation_file = "/home/pdas4/vitcap_r/data/MSCOCO/annotations/captions_val2014.json"
output_dir = "/home/pdas4/vitcap_r/embeddings"
batch_size = 32

os.makedirs(output_dir, exist_ok=True)

# ---------------------
# Load components
# ---------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

vocab = load_vocab(vocab_path)
dataloader = get_loader(
    image_dir=image_dir,
    annotation_file=annotation_file,
    vocab=vocab,
    batch_size=batch_size,
    max_len=35
)

model = DualEncoder(embed_dim=256, vocab_size=len(vocab), hidden_dim=256)
model.load_state_dict(torch.load(model_path, map_location=device))
model = model.to(device)
model.eval()

# ---------------------
# Extract embeddings
# ---------------------
img_embeds = []
txt_embeds = []
hardest_neg_indices = []

with torch.no_grad():
    for images, captions, _ in tqdm(dataloader, desc="Extracting Embeddings"):
        images = torch.stack(images).to(device)
        captions = torch.stack(captions).to(device)

        img_vecs, txt_vecs = model(images, captions)

        img_embeds.append(img_vecs)
        txt_embeds.append(txt_vecs)

        # Compute hard negatives (2nd most similar caption)
        sim_matrix = F.cosine_similarity(img_vecs.unsqueeze(1), txt_vecs.unsqueeze(0), dim=2)
        _, topk_indices = torch.topk(sim_matrix, k=2, dim=1)
        hardest_neg_indices.append(topk_indices[:, 1].cpu())  # 2nd best match

# ---------------------
# Save tensors
# ---------------------
img_tensor = torch.cat(img_embeds, dim=0)
txt_tensor = torch.cat(txt_embeds, dim=0)
neg_tensor = torch.cat(hardest_neg_indices, dim=0)

torch.save(img_tensor, os.path.join(output_dir, "img_embeds.pt"))
torch.save(txt_tensor, os.path.join(output_dir, "txt_embeds.pt"))
torch.save(neg_tensor, os.path.join(output_dir, "hardest_neg_indices.pt"))

print("✅ Embeddings saved to:")
print(f"  - {output_dir}/img_embeds.pt")
print(f"  - {output_dir}/txt_embeds.pt")
print(f"  - {output_dir}/hardest_neg_indices.pt")
