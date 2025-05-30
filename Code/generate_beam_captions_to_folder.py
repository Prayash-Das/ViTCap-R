import os
import torch
from PIL import Image
import matplotlib.pyplot as plt
from torchvision import transforms
from vitcap_model import PatchAttentionDecoder
from utils import load_vocab
from dual_encoder_model import DualEncoder


# Setup
vocab = load_vocab("/home/pdas4/vitcap_r/data/coco_vocab.pkl")
inv_vocab = {idx: word for word, idx in vocab.items()}
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = PatchAttentionDecoder(embed_dim=256, hidden_dim=256, vocab_size=len(vocab)).to(device)
dual_encoder = DualEncoder(embed_dim=256, hidden_dim=256, vocab_size=len(vocab)).to(device)
dual_encoder.load_state_dict(torch.load("/home/pdas4/vitcap_r/models/dual_encoder_best.pth", map_location=device))

# Load model
checkpoint = torch.load("/home/pdas4/vitcap_r/models/patch_attention_best.pth", map_location=device)
filtered = {k.replace('module.', ''): v for k, v in checkpoint.items()}
model.load_state_dict(filtered, strict=False)
model.eval()

# Image preprocessing
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5]*3, std=[0.5]*3)
])

'''# Beam Search
def beam_search(model, image, vocab, device, beam_size=5, max_len=35):
    model.eval()
    image = image.unsqueeze(0).to(device)
    with torch.no_grad():
        vit_out = model.vit(pixel_values=image).last_hidden_state
        patches = model.linear_patch(vit_out)
    sequences = [[[], 0.0, None]]
    for _ in range(max_len):
        all_candidates = []
        for seq, score, hidden in sequences:
            if seq and seq[-1] == vocab['endseq']:
                all_candidates.append((seq, score, hidden))
                continue
            last_tok = vocab['startseq'] if not seq else seq[-1]
            tok_tensor = torch.tensor([[last_tok]], device=device)
            emb = model.embedding(tok_tensor)
            if hidden is None:
                hidden = (torch.zeros(1, 1, model.lstm.hidden_size, device=device),
                          torch.zeros(1, 1, model.lstm.hidden_size, device=device))
            context, _ = model.attention(patches, hidden[0].squeeze(0))
            lstm_input = torch.cat((emb.squeeze(1), context), dim=1).unsqueeze(1)
            output, hidden = model.lstm(lstm_input, hidden)
            logits = model.fc_out(torch.cat((output.squeeze(1), context), dim=1))
            log_probs = torch.log_softmax(logits, dim=1)
            topk_vals, topk_idxs = torch.topk(log_probs, beam_size)
            for i in range(beam_size):
                candidate = (seq + [topk_idxs[0][i].item()], score + topk_vals[0][i].item(), hidden)
                all_candidates.append(candidate)
        sequences = sorted(all_candidates, key=lambda tup: tup[1], reverse=True)[:beam_size]
    return sequences[0][0]'''

def beam_search(model, image, vocab, device, beam_size=5, max_len=35):
    model.eval()
    image = image.unsqueeze(0).to(device)
    with torch.no_grad():
        vit_out = model.vit(pixel_values=image).last_hidden_state
        patches = model.linear_patch(vit_out)
    sequences = [[[], 0.0, None]]
    for _ in range(max_len):
        all_candidates = []
        for seq, score, hidden in sequences:
            if seq and seq[-1] == vocab['endseq']:
                all_candidates.append((seq, score, hidden))
                continue
            last_tok = vocab['startseq'] if not seq else seq[-1]
            tok_tensor = torch.tensor([[last_tok]], device=device)
            emb = model.embedding(tok_tensor)
            if hidden is None:
                hidden = (torch.zeros(1, 1, model.lstm.hidden_size, device=device),
                          torch.zeros(1, 1, model.lstm.hidden_size, device=device))
            context, _ = model.attention(patches, hidden[0].squeeze(0))
            lstm_input = torch.cat((emb.squeeze(1), context), dim=1).unsqueeze(1)
            output, hidden = model.lstm(lstm_input, hidden)
            logits = model.fc_out(torch.cat((output.squeeze(1), context), dim=1))
            log_probs = torch.log_softmax(logits, dim=1)
            topk_vals, topk_idxs = torch.topk(log_probs, beam_size)
            for i in range(beam_size):
                candidate = (seq + [topk_idxs[0][i].item()], score + topk_vals[0][i].item(), hidden)
                all_candidates.append(candidate)
        sequences = sorted(all_candidates, key=lambda tup: tup[1], reverse=True)[:beam_size]
    return [seq for seq, _, _ in sequences]  # ⬅️ Return all candidates


# Decode
def decode(tokens, inv_vocab):
    return ' '.join([inv_vocab[tok] for tok in tokens if inv_vocab[tok] not in {'<pad>', 'startseq', 'endseq'}])

def rerank(image_tensor, beam_candidates):
    with torch.no_grad():
        img_vec = dual_encoder.forward_image(image_tensor.unsqueeze(0).to(device))
        sims = []
        for seq in beam_candidates:
            cap_tensor = torch.tensor(seq).unsqueeze(0).to(device)
            txt_vec = dual_encoder.forward_text(cap_tensor)
            sim = torch.nn.functional.cosine_similarity(img_vec, txt_vec).item()
            sims.append(sim)
    best_idx = int(torch.tensor(sims).argmax())
    return beam_candidates[best_idx]




# Run for 10 images
img_dir = "/home/pdas4/vitcap_r/data/MSCOCO/val2014"
save_dir = "/home/pdas4/vitcap_r/output_captions_rerank"
os.makedirs(save_dir, exist_ok=True)

img_files = sorted([f for f in os.listdir(img_dir) if f.endswith(".jpg")])[:10]

for idx, f in enumerate(img_files):
    path = os.path.join(img_dir, f)
    image = Image.open(path).convert('RGB')
    tensor = transform(image)
   # tokens = beam_search(model, tensor, vocab, device)
    tokens_all = beam_search(model, tensor, vocab, device)
    tokens = rerank(tensor, tokens_all)
    caption = decode(tokens, inv_vocab)

    # Save plot
    plt.figure(figsize=(6, 5))
    plt.imshow(image)
    plt.title(caption, fontsize=10)
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"caption_{idx+1}.png"), dpi=300)
    plt.close()

print(f"✅ Captions saved to: {save_dir}")
