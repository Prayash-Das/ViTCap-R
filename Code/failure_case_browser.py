import torch
import os
from utils import load_vocab
from pycocotools.coco import COCO
from PIL import Image
import matplotlib.pyplot as plt
import torch.nn.functional as F

# -----------------------------
# Paths
# -----------------------------
image_dir = "/home/pdas4/vitcap_r/data/MSCOCO/val2014"
ann_file = "/home/pdas4/vitcap_r/data/MSCOCO/annotations/captions_val2014.json"
vocab_path = "/home/pdas4/vitcap_r/data/coco_vocab.pkl"

# Embeddings
img_tensor = torch.load("/home/pdas4/vitcap_r/embeddings/img_embeds.pt")
txt_tensor = torch.load("/home/pdas4/vitcap_r/embeddings/txt_embeds.pt")
hard_neg_indices = torch.load("/home/pdas4/vitcap_r/embeddings/hardest_neg_indices.pt")

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
img_tensor = img_tensor.to(device)
txt_tensor = txt_tensor.to(device)
hard_neg_indices = hard_neg_indices.to(device)

# Load COCO
coco = COCO(ann_file)
img_ids = list(coco.imgs.keys())

# Compute similarity
hard_neg_txt = txt_tensor[hard_neg_indices]
pos_scores = F.cosine_similarity(img_tensor, txt_tensor)
neg_scores = F.cosine_similarity(img_tensor, hard_neg_txt)

# Find worst retrievals (smallest margin)
margins = pos_scores - neg_scores
worst_indices = torch.topk(margins, k=10, largest=False).indices  # 10 worst

# -----------------------------
# Show & Save Results
# -----------------------------
output_dir = "/home/pdas4/vitcap_r/failures"
os.makedirs(output_dir, exist_ok=True)

for i, idx in enumerate(worst_indices.tolist()):
    img_id = img_ids[idx]
    img_file = coco.loadImgs(img_id)[0]['file_name']
    img_path = os.path.join(image_dir, img_file)

    # Load image
    image = Image.open(img_path).convert('RGB')

    # Load captions
    ann_ids = coco.getAnnIds(imgIds=img_id)
    captions = coco.loadAnns(ann_ids)
    gt_caption = captions[0]['caption'] if captions else "[Missing caption]"

    # Get hard negative caption (as string)
    neg_caption_idx = hard_neg_indices[idx].item()
    neg_ann_ids = coco.getAnnIds(imgIds=img_ids[neg_caption_idx])
    neg_caps = coco.loadAnns(neg_ann_ids)
    neg_caption = neg_caps[0]['caption'] if neg_caps else "[Missing neg caption]"

    # Plot and annotate
    plt.figure(figsize=(6, 5))
    plt.imshow(image)
    plt.axis("off")
    plt.title(f"❌ Hard Negative vs ✅ True", fontsize=10)
    plt.figtext(0.5, 0.01, f"❌ {neg_caption}", wrap=True, ha='center', color='red')
    plt.figtext(0.5, 0.07, f"✅ {gt_caption}", wrap=True, ha='center', color='green')

    save_path = os.path.join(output_dir, f"failure_{i+1}.png")
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()

print(f"✅ Saved top-10 failure visualizations to: {output_dir}")
