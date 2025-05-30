import torch
from transformers import ViTFeatureExtractor
from nltk.tokenize import TreebankWordTokenizer
from PIL import Image
import torchvision.transforms as T
import pickle

from vitcap_model import PatchAttentionDecoder  # import your model class

# ---- CONFIG ----
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMG_PATH = "/home/pdas4/vitcap_r/Data/MSCOCO/val2014/COCO_val2014_000000000074.jpg"
VOCAB_PATH = "coco_vocab.pkl"
MODEL_PATH = "/home/pdas4/vitcap_r/models/patch_attention_best.pth"
MAX_LEN = 35

# ---- Load Vocabulary ----
with open(VOCAB_PATH, "rb") as f:
    vocab = pickle.load(f)
inv_vocab = {v: k for k, v in vocab.items()}

# ---- Load Model ----
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_path = "/home/pdas4/vitcap_r/models/patch_attention_best.pth"
model = PatchAttentionDecoder(embed_dim=256, hidden_dim=256, vocab_size=len(vocab))
checkpoint = torch.load(model_path, map_location=device)
filtered_checkpoint = {k: v for k, v in checkpoint.items() if not k.startswith("vit.")}
model.load_state_dict(filtered_checkpoint, strict=False)
model.to(DEVICE).eval()

# ---- Image Preprocessing ----
transform = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize([0.5]*3, [0.5]*3)
])
img = Image.open(IMG_PATH).convert("RGB")
img_tensor = transform(img).unsqueeze(0).to(DEVICE)

# ---- Generate Caption (Greedy) ----
tokenizer = TreebankWordTokenizer()
caption = ["startseq"]
with torch.no_grad():
    vit_out = model.vit(pixel_values=img_tensor).last_hidden_state
    patch_feats = model.linear_patch(vit_out)
    hidden = None

    for _ in range(MAX_LEN):
        inputs = torch.tensor([[vocab.get(c, vocab['<unk>']) for c in caption]], device=DEVICE)
        embeds = model.embedding(inputs[:, -1])
        context, _ = model.attention(patch_feats, hidden[0].squeeze(0) if hidden else torch.zeros(1, patch_feats.size(2)).to(DEVICE))
        lstm_input = torch.cat((embeds, context), dim=-1).unsqueeze(1)
        output, hidden = model.lstm(lstm_input, hidden)
        output = torch.cat((output.squeeze(1), context), dim=-1)
        output = model.fc_out(output)
        next_token = output.argmax(-1).item()
        next_word = inv_vocab.get(next_token, "<unk>")
        if next_word == "endseq":
            break
        caption.append(next_word)

print("📝 Caption:", " ".join(caption[1:]))
