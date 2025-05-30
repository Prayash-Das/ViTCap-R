import os
import torch
from tqdm import tqdm
from pycocotools.coco import COCO
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.translate.meteor_score import meteor_score
from nltk.tokenize import TreebankWordTokenizer
from torchvision import transforms
from PIL import Image

from vitcap_model import PatchAttentionDecoder
from utils import load_vocab
from beam_decoder import beam_search_decoding, decode_caption

tokenizer = TreebankWordTokenizer()

# Paths
image_dir = "/home/pdas4/vitcap_r/data/MSCOCO/val2014"
ann_file = "/home/pdas4/vitcap_r/data/MSCOCO/annotations/captions_val2014.json"
vocab_path = "/home/pdas4/vitcap_r/data/coco_vocab.pkl"
model_path = "/home/pdas4/vitcap_r/models/patch_attention_best.pth"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load vocab and inverse vocab
vocab = load_vocab(vocab_path)
inv_vocab = {idx: word for word, idx in vocab.items()}

# Load model
model = PatchAttentionDecoder(embed_dim=256, hidden_dim=256, vocab_size=len(vocab))
checkpoint = torch.load(model_path, map_location=device)
filtered_checkpoint = {k: v for k, v in checkpoint.items() if not k.startswith("vit.")}
model.load_state_dict(filtered_checkpoint, strict=False)
model.to(device).eval()

# Image transform
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5]*3, std=[0.5]*3)
])

# Load COCO
coco = COCO(ann_file)
img_ids = list(coco.imgs.keys())[:2000]  # Evaluate on first 100 images

# Evaluation metrics
smooth = SmoothingFunction().method1
bleu1_scores, bleu2_scores, bleu3_scores, bleu4_scores, meteor_scores = [], [], [], [], []

for img_id in tqdm(img_ids, desc="Evaluating"):
    meta = coco.loadImgs(img_id)[0]
    image_path = os.path.join(image_dir, meta['file_name'])
    image = Image.open(image_path).convert('RGB')
    tensor = transform(image).to(device)

    tokens = beam_search_decoding(model, tensor, vocab, device, beam_size=5)
    hyp_caption = decode_caption(tokens, inv_vocab)
    hyp_tokens = tokenizer.tokenize(hyp_caption.lower())

    ann_ids = coco.getAnnIds(imgIds=img_id)
    anns = coco.loadAnns(ann_ids)
    ref_tokens = [tokenizer.tokenize(ann['caption'].lower()) for ann in anns]

    bleu1_scores.append(sentence_bleu(ref_tokens, hyp_tokens, weights=(1, 0, 0, 0), smoothing_function=smooth))
    bleu2_scores.append(sentence_bleu(ref_tokens, hyp_tokens, weights=(0.5, 0.5, 0, 0), smoothing_function=smooth))
    bleu3_scores.append(sentence_bleu(ref_tokens, hyp_tokens, weights=(0.33, 0.33, 0.33, 0), smoothing_function=smooth))
    bleu4_scores.append(sentence_bleu(ref_tokens, hyp_tokens, weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=smooth))
    meteor_scores.append(meteor_score(ref_tokens, hyp_tokens))

# Print final metrics
print("🎯 BLEU & METEOR Evaluation (avg over 100 samples):")
print(f"BLEU-1: {sum(bleu1_scores)/len(bleu1_scores):.4f}")
print(f"BLEU-2: {sum(bleu2_scores)/len(bleu2_scores):.4f}")
print(f"BLEU-3: {sum(bleu3_scores)/len(bleu3_scores):.4f}")
print(f"BLEU-4: {sum(bleu4_scores)/len(bleu4_scores):.4f}")
print(f"METEOR : {sum(meteor_scores)/len(meteor_scores):.4f}")
