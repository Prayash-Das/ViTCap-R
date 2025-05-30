import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import ViTModel

class DualEncoder(nn.Module):
    def __init__(self, embed_dim, vocab_size, hidden_dim):
        super().__init__()
        self.image_encoder = ViTModel.from_pretrained('google/vit-base-patch16-224-in21k')
        self.linear_proj = nn.Linear(768, embed_dim)
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True)
        self.text_proj = nn.Linear(hidden_dim, embed_dim)

    def forward_image(self, images):
        with torch.no_grad():
            vit_out = self.image_encoder(pixel_values=images).last_hidden_state[:, 0, :]
        img_embed = self.linear_proj(vit_out)
        return F.normalize(img_embed, dim=1)

    def forward_text(self, captions):
        embeds = self.embedding(captions)
        _, (h_n, _) = self.lstm(embeds)
        txt_embed = self.text_proj(h_n[-1])
        return F.normalize(txt_embed, dim=1)

    def forward(self, images, captions):
        return self.forward_image(images), self.forward_text(captions)

def contrastive_loss(img_vecs, txt_vecs, temperature=0.07):
    logits = torch.matmul(img_vecs, txt_vecs.T) / temperature
    labels = torch.arange(len(img_vecs)).to(img_vecs.device)
    loss_i2t = F.cross_entropy(logits, labels)
    loss_t2i = F.cross_entropy(logits.T, labels)
    return (loss_i2t + loss_t2i) / 2
