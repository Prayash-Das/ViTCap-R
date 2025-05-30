import torch
import os
import matplotlib.pyplot as plt
import numpy as np
from torchvision import transforms
from PIL import Image
from vitcap_model import PatchAttentionDecoder
from utils import load_vocab


def generate_caption_with_attention(model, image, vocab, device, max_len=35):
    model.eval()
    with torch.no_grad():
        vit_out = model.vit(pixel_values=image.unsqueeze(0)).last_hidden_state
        patch_feats = model.linear_patch(vit_out)

    token = vocab['startseq']
    tokens = []
    attn_weights = []
    hidden = None

    for _ in range(max_len):
        input_token = torch.tensor([[token]], device=device)
        embedded = model.embedding(input_token)

        if hidden is None:
            h_size = model.lstm.hidden_size
            hidden = (
                torch.zeros(1, 1, h_size, device=device),
                torch.zeros(1, 1, h_size, device=device)
            )

        context, alpha = model.attention(patch_feats, hidden[0].squeeze(0))
        lstm_input = torch.cat((embedded.squeeze(1), context), dim=1).unsqueeze(1)
        output, hidden = model.lstm(lstm_input, hidden)
        output = torch.cat((output.squeeze(1), context), dim=1)
        logits = model.fc_out(output)

        token = torch.argmax(logits, dim=1).item()
        tokens.append(token)
        attn_weights.append(alpha.squeeze(0).cpu())

        if token == vocab['endseq']:
            break

    return tokens, attn_weights



# ---------------------
# Visualization Helpers
# ---------------------

def visualize_full_attention_grid(image, attention_tensor, words, save_path="attention_grid.png"):
    import torchvision.transforms.functional as TF

    image = TF.resize(image, (224, 224))
    num_words = len(attention_tensor)
    cols = 5
    rows = int(np.ceil(num_words / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(15, rows * 3))

    for i, (attn, word) in enumerate(zip(attention_tensor, words)):
        if isinstance(attn, torch.Tensor):
            attn = attn.detach().cpu()

        if attn.numel() == 0:
            print(f"⚠️ Skipping empty attention for word: {word}")
            continue

        attn_map = attn[1:].reshape(14, 14).numpy()

        row, col = divmod(i, cols)
        ax = axes[row][col] if rows > 1 else axes[col]
        ax.imshow(image)
        ax.imshow(attn_map, cmap='jet', alpha=0.5)
        ax.set_title(word)
        ax.axis('off')

    for j in range(i + 1, rows * cols):
        row, col = divmod(j, cols)
        ax = axes[row][col] if rows > 1 else axes[col]
        ax.axis('off')

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"📸 Saved attention grid to: {save_path}")

# ---------------------
# Main Script
# ---------------------

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
vocab = load_vocab("/home/pdas4/vitcap_r/data/coco_vocab.pkl")
idx2word = {i: w for w, i in vocab.items()}

model = PatchAttentionDecoder(embed_dim=256, hidden_dim=256, vocab_size=len(vocab))
checkpoint = torch.load("/home/pdas4/vitcap_r/models/patch_attention_best.pth", map_location=device)
filtered_ckpt = {k: v for k, v in checkpoint.items() if not k.startswith("vit.")}
model.load_state_dict(filtered_ckpt, strict=False)
model.to(device)

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5]*3, std=[0.5]*3)
])

img_path = "/home/pdas4/vitcap_r/data/MSCOCO/val2014/COCO_val2014_000000000488.jpg"
image_pil = Image.open(img_path).convert("RGB")
image_tensor = transform(image_pil).to(device)

tokens, attn_weights = generate_caption_with_attention(model, image_tensor, vocab, device)

# 🔍 Debugging
print(f"🔍 attn_weights shape: {len(attn_weights)}")

# Convert to words and clean
words = [idx2word.get(t, '<unk>') for t in tokens]
attn_clean = [a for a, w in zip(attn_weights, words) if w not in {'<pad>', 'startseq', 'endseq'} and a.numel() > 0]
clean_words = [w for w in words if w not in {'<pad>', 'startseq', 'endseq'}]

# Output folder
os.makedirs("attention_maps", exist_ok=True)
grid_path = "attention_maps/full_attention_grid.png"
visualize_full_attention_grid(image_pil, attn_clean, clean_words, save_path=grid_path)
