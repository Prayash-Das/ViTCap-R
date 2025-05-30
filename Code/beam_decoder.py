import torch
from utils import load_vocab
#from dual_encoder_model import DualEncoder
from vitcap_model import PatchAttentionDecoder
def beam_search_decoding(model, image, vocab, device, beam_size=3, max_len=35):
    model.eval()
    image = image.unsqueeze(0).to(device)  # Add batch dimension
    with torch.no_grad():
        vit_out = model.vit(pixel_values=image).last_hidden_state
        patch_feats = model.linear_patch(vit_out)

    # Start the beam
    sequences = [[list(), 0.0, None]]  # (tokens, score, hidden)

    for _ in range(max_len):
        all_candidates = []
        for seq, score, hidden in sequences:
            if len(seq) > 0 and seq[-1] == vocab['endseq']:
                all_candidates.append((seq, score, hidden))
                continue

            # Prepare input
            if len(seq) == 0:
                input_token = torch.tensor([[vocab['startseq']]], device=device)
            else:
                input_token = torch.tensor([[seq[-1]]], device=device)

            embeddings = model.embedding(input_token)
            if hidden is None:
                hidden = (torch.zeros(1, 1, model.lstm.hidden_size, device=device),
                          torch.zeros(1, 1, model.lstm.hidden_size, device=device))

            context, alpha = model.attention(patch_feats, hidden[0].squeeze(0))
            lstm_input = torch.cat((embeddings.squeeze(1), context), dim=1).unsqueeze(1)
            output, hidden = model.lstm(lstm_input, hidden)
            output = torch.cat((output.squeeze(1), context), dim=1)
            output = model.fc_out(output)

            log_probs = torch.log_softmax(output, dim=1)

            topk_log_probs, topk_tokens = torch.topk(log_probs, beam_size)

            for i in range(beam_size):
                candidate = (seq + [topk_tokens[0][i].item()], score + topk_log_probs[0][i].item(), hidden)
                all_candidates.append(candidate)

        # Select top beam_size sequences
        ordered = sorted(all_candidates, key=lambda tup: tup[1], reverse=True)
        sequences = ordered[:beam_size]

    # Select the best sequence
    final_seq = sequences[0][0]

    return final_seq

def decode_caption(tokens, inv_vocab):
    words = []
    for token in tokens:
        word = inv_vocab.get(token, '<unk>')
        if word == 'endseq':
            break
        if word != 'startseq':
            words.append(word)
    return ' '.join(words)


# Load vocab and inverse vocab
vocab = load_vocab("coco_vocab.pkl")
inv_vocab = {idx: word for word, idx in vocab.items()}

# Load model
model = PatchAttentionDecoder(embed_dim=256, hidden_dim=256, vocab_size=len(vocab))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# 💾 Load checkpoint and REMOVE 'module.' prefixes
checkpoint = torch.load("/home/pdas4/vitcap_r/models/patch_attention_best.pth", map_location=device)
new_state_dict = {}
for k, v in checkpoint.items():
    if k.startswith('module.'):
        new_state_dict[k[7:]] = v  # strip 'module.' prefix
    else:
        new_state_dict[k] = v

model.load_state_dict(new_state_dict)
model.eval()

# Load an image from val2014
from torchvision import transforms
from PIL import Image

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5]*3, std=[0.5]*3)
])

image_path = "/home/pdas4/vitcap_r/data/MSCOCO/val2014/COCO_val2014_000000000488.jpg"  # Example
image = Image.open(image_path).convert('RGB')
image = transform(image)

# Generate Caption
tokens = beam_search_decoding(model, image, vocab, device, beam_size=5)
caption = decode_caption(tokens, inv_vocab)

print(f"🖼️ Caption: {caption}")
