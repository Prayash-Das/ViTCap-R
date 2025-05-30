import torch
import torchvision.transforms as T
import json
from PIL import Image
from tqdm import tqdm
from pycocotools.coco import COCO
import pickle
import os
from transformers import ViTModel
from vitcap_model import PatchAttentionDecoder

# ---- CONFIG ----
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMAGE_DIR = "/home/pdas4/vitcap_r/data/MSCOCO/val2014"
ANNOTATION_FILE = "/home/pdas4/vitcap_r/data/MSCOCO/annotations/captions_val2014.json"
VOCAB_PATH = "/home/pdas4/vitcap_r/data/coco_vocab.pkl"
MODEL_PATH = "/home/pdas4/vitcap_r/models/patch_attention_best.pth"
BEAM_WIDTH = 3
MAX_LEN = 35
OUTPUT_JSON = "generated_captions_beam_100.json"

# ---- Load Vocab ----
with open(VOCAB_PATH, "rb") as f:
    vocab = pickle.load(f)
inv_vocab = {v: k for k, v in vocab.items()}

# ---- Load Model ----
# Manually load ViT backbone before loading your weights
vit_model = ViTModel.from_pretrained('google/vit-base-patch16-224-in21k')
model = PatchAttentionDecoder(embed_dim=256, hidden_dim=256, vocab_size=len(vocab))
model.vit = vit_model  # inject ViT backbone
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
# ---- Preprocessing ----
transform = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize([0.5]*3, [0.5]*3)
])

# ---- Beam Search Captioning ----
def beam_search(model, img_tensor, vocab, inv_vocab, beam_width=3, max_len=35):
    with torch.no_grad():
        vit_out = model.vit(pixel_values=img_tensor).last_hidden_state
        patch_feats = model.linear_patch(vit_out)

        start_token = vocab["startseq"]
        end_token = vocab["endseq"]

        sequences = [[start_token]]
        scores = [0.0]
        hidden_states = [None]

        for _ in range(max_len):
            all_candidates = []
            for seq, score, hidden in zip(sequences, scores, hidden_states):
                if seq[-1] == end_token:
                    all_candidates.append((seq, score, hidden))
                    continue

                last_token = torch.tensor([[seq[-1]]], device=DEVICE)
                embed = model.embedding(last_token)

                context, _ = model.attention(patch_feats, hidden[0].squeeze(0) if hidden else torch.zeros(1, patch_feats.size(2)).to(DEVICE))
                lstm_input = torch.cat((embed, context), dim=-1).unsqueeze(1)
                output, new_hidden = model.lstm(lstm_input, hidden)
                output = torch.cat((output.squeeze(1), context), dim=-1)
                logits = model.fc_out(output)
                log_probs = torch.log_softmax(logits, dim=-1).squeeze(0)

                topk_log_probs, topk_indices = torch.topk(log_probs, beam_width)
                for i in range(beam_width):
                    candidate = seq + [topk_indices[i].item()]
                    candidate_score = score + topk_log_probs[i].item()
                    all_candidates.append((candidate, candidate_score, new_hidden))

            ordered = sorted(all_candidates, key=lambda tup: tup[1], reverse=True)
            sequences, scores, hidden_states = zip(*ordered[:beam_width])

        final_tokens = sequences[0]
        caption = [inv_vocab.get(tok, "<unk>") for tok in final_tokens if tok not in [start_token, end_token, vocab["<pad>"]]]
        return " ".join(caption)

# ---- Load COCO Dataset ----
coco = COCO(ANNOTATION_FILE)
img_ids = coco.getImgIds()[:100]  # 💯 Subset of 100 images
results = []

for img_id in tqdm(img_ids, desc="🖼️ Generating Beam Captions"):
    path = coco.loadImgs(img_id)[0]['file_name']
    image = Image.open(os.path.join(IMAGE_DIR, path)).convert('RGB')
    img_tensor = transform(image).unsqueeze(0).to(DEVICE)
    caption = beam_search(model, img_tensor, vocab, inv_vocab, beam_width=BEAM_WIDTH)
    results.append({"image_id": img_id, "caption": caption})

# ---- Save Results ----
with open(OUTPUT_JSON, "w") as f:
    json.dump(results, f)
print(f"✅ Saved beam search captions to {OUTPUT_JSON}")
