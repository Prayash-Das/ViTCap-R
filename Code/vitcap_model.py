from transformers import ViTModel
import torch
import torch.nn as nn
import torch.nn.functional as F

class PatchAttentionDecoder(nn.Module):
    def __init__(self, embed_dim, hidden_dim, vocab_size, patch_count=197):
        super().__init__()
        self.vit = ViTModel.from_pretrained('google/vit-base-patch16-224-in21k')
        self.linear_patch = nn.Linear(768, embed_dim)
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(512, hidden_dim, batch_first=True)
        self.attn_fc1 = nn.Linear(embed_dim + hidden_dim, 256)
        self.attn_fc2 = nn.Linear(256, 1)
        self.fc_out = nn.Linear(hidden_dim + embed_dim, vocab_size)

    def attention(self, features, hidden_state):
        hidden_state = hidden_state.unsqueeze(1).repeat(1, features.size(1), 1)
        concat = torch.cat((features, hidden_state), dim=2)
        energy = torch.tanh(self.attn_fc1(concat))
        attention = self.attn_fc2(energy).squeeze(2)
        weights = F.softmax(attention, dim=1)
        context = torch.bmm(weights.unsqueeze(1), features).squeeze(1)
        return context, weights

    def forward(self, images, captions):
        with torch.no_grad():
            vit_out = self.vit(pixel_values=images).last_hidden_state
        patch_feats = self.linear_patch(vit_out)
        captions_embeds = self.embedding(captions[:, :-1])
        batch_size, seq_len, _ = captions_embeds.size()
        hidden, outputs, attn_weights = None, [], []
        for t in range(seq_len):
            context, alpha = self.attention(patch_feats, hidden[0].squeeze(0) if hidden else torch.zeros(batch_size, patch_feats.size(2)).to(images.device))
            lstm_input = torch.cat((captions_embeds[:, t, :], context), dim=1).unsqueeze(1)
            output, hidden = self.lstm(lstm_input, hidden)
            output = torch.cat((output.squeeze(1), context), dim=1)
            output = self.fc_out(output)
            outputs.append(output.unsqueeze(1))
            attn_weights.append(alpha.unsqueeze(1))
        outputs = torch.cat(outputs, dim=1)
        attn_weights = torch.cat(attn_weights, dim=1)
        return outputs, attn_weights
