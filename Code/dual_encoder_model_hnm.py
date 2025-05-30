import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import ViTModel

class DualEncoder(nn.Module):
    def __init__(self, embed_dim, vocab_size, hidden_dim=256):
        super(DualEncoder, self).__init__()

        # Image encoder (frozen ViT)
        self.vit = ViTModel.from_pretrained('google/vit-base-patch16-224-in21k')
        for param in self.vit.parameters():
            param.requires_grad = False
        self.linear_image = nn.Linear(768, embed_dim)

        # Text encoder (LSTM)
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True)
        self.linear_text = nn.Linear(hidden_dim, embed_dim)

    def forward(self, images, captions):
        img_embeds = self.forward_image(images)
        txt_embeds = self.forward_text(captions)
        return img_embeds, txt_embeds

    def forward_image(self, images):
        with torch.no_grad():
            vit_feats = self.vit(pixel_values=images).last_hidden_state[:, 0, :]  # CLS token
        return self.linear_image(vit_feats)  # [B, embed_dim]

    def forward_text(self, captions):
        embeds = self.embedding(captions)  # [B, T, embed_dim]
        _, (hidden, _) = self.lstm(embeds)  # hidden: [1, B, hidden_dim]
        return self.linear_text(hidden.squeeze(0))  # [B, embed_dim]


def contrastive_loss(image_embeddings, text_embeddings, temperature=0.07):
    """
    Symmetric InfoNCE loss with in-batch hard negatives.
    """
    # Normalize embeddings
    image_embeddings = F.normalize(image_embeddings, dim=1)
    text_embeddings = F.normalize(text_embeddings, dim=1)

    # Cosine similarity matrix
    logits = image_embeddings @ text_embeddings.T / temperature  # [B, B]
    batch_size = logits.size(0)
    labels = torch.arange(batch_size).to(image_embeddings.device)

    # Loss for both directions
    loss_i2t = F.cross_entropy(logits, labels)
    loss_t2i = F.cross_entropy(logits.T, labels)

    return (loss_i2t + loss_t2i) / 2
