import torch
from PIL import Image
from torchvision import transforms
from pycocotools.coco import COCO
from vitcap_model import PatchAttentionDecoder
from utils import load_vocab
import heapq
import os
import matplotlib.pyplot as plt
os.makedirs("output_captions", exist_ok=True)
# ----------------------------
# Paths and Setup
# ----------------------------
image_dir = "/home/pdas4/vitcap_r/data/MSCOCO/val2014"
ann_file = "/home/pdas4/vitcap_r/data/MSCOCO/annotations/captions_val2014.json"
vocab_path = "/home/pdas4/vitcap_r/data/coco_vocab.pkl"
model_path = "/home/pdas4/vitcap_r/models/patch_attention_best.pth"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load vocab and model
vocab = load_vocab(vocab_path)
idx2word = {i: w for w, i in vocab.items()}
model = PatchAttentionDecoder(embed_dim=256, hidden_dim=256, vocab_size=len(vocab))
checkpoint = torch.load(model_path, map_location=device)

filtered_checkpoint = {k: v for k, v in checkpoint.items() if not k.startswith("vit.")}
model.load_state_dict(filtered_checkpoint, strict=False)
model.to(device).eval()

# Transform
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5]*3, std=[0.5]*3)
])

# Beam Search
def beam_search_caption(model, image, vocab, beam_width=5, max_len=35):
    with torch.no_grad():
        vit_out = model.vit(pixel_values=image.unsqueeze(0)).last_hidden_state
        patch_feats = model.linear_patch(vit_out)
        k = beam_width
        sequences = [[list(), 0.0, None]]

        for _ in range(max_len):
            all_candidates = []
            for seq, score, hidden in sequences:
                if seq and seq[-1] == vocab['endseq']:
                    all_candidates.append((seq, score, hidden))
                    continue

                last_token = vocab['startseq'] if not seq else seq[-1]
                embedded = model.embedding(torch.tensor([last_token]).to(image.device)).unsqueeze(0)
                context, _ = model.attention(patch_feats, hidden[0].squeeze(0) if hidden else torch.zeros(1, patch_feats.size(2)).to(image.device))
                lstm_input = torch.cat((embedded, context.unsqueeze(1)), dim=2)
                output, new_hidden = model.lstm(lstm_input, hidden)
                output = torch.cat((output.squeeze(1), context), dim=1)
                logits = model.fc_out(output)
                log_probs = torch.nn.functional.log_softmax(logits, dim=1)

                topk = torch.topk(log_probs, k)
                for i in range(k):
                    token = topk.indices[0][i].item()
                    prob = topk.values[0][i].item()
                    candidate = (seq + [token], score + prob, new_hidden)
                    all_candidates.append(candidate)

            sequences = heapq.nlargest(k, all_candidates, key=lambda tup: tup[1])

        return [' '.join([idx2word[i] for i in s if i not in {vocab['startseq'], vocab['endseq'], vocab['<pad>']}]) for s, _, _ in sequences]

# ----------------------------
# Show Images and Captions
# ----------------------------
coco = COCO(ann_file)
img_ids = list(coco.imgs.keys())[:5]

for img_id in img_ids:
    meta = coco.loadImgs(img_id)[0]
    image_path = os.path.join(image_dir, meta['file_name'])
    image = Image.open(image_path).convert('RGB')
    tensor = transform(image).to(device)

    captions = beam_search_caption(model, tensor, vocab, beam_width=5)

    # Plot image + captions
    plt.figure(figsize=(8, 6))
    plt.imshow(image)
    plt.axis('off')
    cap_text = "\n".join([f"{i+1}. {cap}" for i, cap in enumerate(captions)])
    plt.title(f"{meta['file_name']}\n\n{cap_text}", fontsize=10)
    save_path = os.path.join("output_captions", f"{meta['file_name'].split('.')[0]}_caption.png")
    plt.savefig(save_path, dpi=300)
    print(f" Saved: {save_path}")
    plt.close()
