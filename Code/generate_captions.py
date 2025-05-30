import torch
from torch.utils.data import DataLoader
from transformers import ViTModel
from torchvision import transforms
from PIL import Image
from pycocotools.coco import COCO
import os
import json
from tqdm import tqdm
from nltk.tokenize import TreebankWordTokenizer
from vitcap_model  import PatchAttentionDecoder  # make sure your model class is accessible
from utils import load_vocab

tokenizer = TreebankWordTokenizer()

# -------------------------
# 🧠 Caption Generator
# -------------------------
def generate_caption(model, image_tensor, vocab, max_len=35):
    model.eval()
    with torch.no_grad():
        vit_out = model.vit(pixel_values=image_tensor.unsqueeze(0)).last_hidden_state
        patch_feats = model.linear_patch(vit_out)

        inputs = torch.tensor([[vocab['startseq']]]).to(image_tensor.device)
        hidden = None
        caption = []

        for _ in range(max_len):
            embed = model.embedding(inputs[:, -1])
            context, _ = model.attention(patch_feats, hidden[0].squeeze(0) if hidden else torch.zeros(1, patch_feats.size(2)).to(image_tensor.device))
            lstm_input = torch.cat((embed, context), dim=1).unsqueeze(1)
            output, hidden = model.lstm(lstm_input, hidden)
            output = torch.cat((output.squeeze(1), context), dim=1)
            logits = model.fc_out(output)
            predicted_id = logits.argmax(-1).item()

            if predicted_id == vocab['endseq']:
                break
            caption.append(predicted_id)
            inputs = torch.cat([inputs, torch.tensor([[predicted_id]]).to(image_tensor.device)], dim=1)

        # Reverse vocab
        inv_vocab = {idx: tok for tok, idx in vocab.items()}
        return ' '.join([inv_vocab.get(idx, '') for idx in caption])

# -------------------------
# 🖼️ Load Dataset
# -------------------------
def load_val_images(image_dir, annotation_file, transform):
    coco = COCO(annotation_file)
    ids = list(coco.imgs.keys())
    data = []
    for img_id in ids:
        filename = coco.loadImgs(img_id)[0]['file_name']
        img_path = os.path.join(image_dir, filename)
        image = Image.open(img_path).convert('RGB')
        image_tensor = transform(image)
        data.append((img_id, image_tensor))
    return data

# -------------------------
# 🔧 Main
# -------------------------
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vocab = load_vocab("/home/pdas4/vitcap_r/data/coco_vocab.pkl")

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5]*3, std=[0.5]*3)
    ])

    val_data = load_val_images(
        image_dir="/home/pdas4/vitcap_r/data/MSCOCO/val2014",
        annotation_file="/home/pdas4/vitcap_r/data/MSCOCO/annotations/captions_val2014.json",
        transform=transform
    )

    model = PatchAttentionDecoder(embed_dim=256, hidden_dim=256, vocab_size=len(vocab))
    model.to(device)
    
    checkpoint = torch.load("/home/pdas4/vitcap_r/models/patch_attention_best.pth", map_location=device)
    model.load_state_dict(checkpoint, strict=False)
    results = []
    for img_id, img_tensor in tqdm(val_data[:1000], desc="🔍 Generating Captions"):  # limit to 1000 for speed
        img_tensor = img_tensor.to(device)
        caption = generate_caption(model, img_tensor, vocab)
        results.append({"image_id": img_id, "caption": caption})

    # Save to JSON
    with open("generated_captions.json", "w") as f:
        json.dump(results, f)

    print("✅ Saved 1000 generated captions to generated_captions.json")
