#!/usr/bin/env python
# coding: utf-8

# In[1]:


#Preprocessing: Build Vocabulary


# In[6]:

import torch
import torch.nn as nn
import torch.nn.functional as F
from pycocotools.coco import COCO
import nltk
from nltk.tokenize import TreebankWordTokenizer
tokenizer = TreebankWordTokenizer()
from collections import Counter
import json
from torch.utils.data import DataLoader


def build_vocab(annotation_file, threshold=10):
    coco = COCO(annotation_file)
    counter = Counter()
    for ann_id in coco.anns:
        caption = coco.anns[ann_id]['caption'].lower()
        tokens = tokenizer.tokenize(caption)
        counter.update(tokens)

    # Filter rare words
    words = [word for word in counter if counter[word] >= threshold]
    vocab = {'<pad>': 0, '<unk>': 1, 'startseq': 2, 'endseq': 3}
    for i, word in enumerate(words, start=4):
        vocab[word] = i

    print(f"✅ Vocabulary size: {len(vocab)}")
    return vocab


# In[5]:


import os
from PIL import Image
from torch.utils.data import Dataset
import torch
from torchvision import transforms

class CocoDataset(Dataset):
    def __init__(self, image_dir, annotation_file, vocab, transform=None, max_len=35):
        self.coco = COCO(annotation_file)
        self.ids = list(self.coco.imgs.keys())
        self.image_dir = image_dir
        self.vocab = vocab
        self.transform = transform or transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5]*3, std=[0.5]*3)
        ])
        self.max_len = max_len

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        img_id = self.ids[idx]
        ann_ids = self.coco.getAnnIds(imgIds=img_id)
        captions = self.coco.loadAnns(ann_ids)
        caption = captions[0]['caption']

        path = self.coco.loadImgs(img_id)[0]['file_name']
        image = Image.open(os.path.join(self.image_dir, path)).convert('RGB')
        image = self.transform(image)

        # Preprocess caption
        tokens = ['startseq'] + tokenizer.tokenize(caption.lower()) + ['endseq']
        caption_ids = [self.vocab.get(tok, self.vocab['<unk>']) for tok in tokens]
        caption_ids = caption_ids[:self.max_len]
        caption_ids += [self.vocab['<pad>']] * (self.max_len - len(caption_ids))

        return image, torch.tensor(caption_ids), img_id


# In[7]:


def get_loader(image_dir, annotation_file, vocab, batch_size=32, max_len=35):
    dataset = CocoDataset(image_dir, annotation_file, vocab, max_len=max_len)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=lambda x: tuple(zip(*x)))
    return loader


# In[9]:


import pickle

def save_vocab(vocab, file_path="coco_vocab.pkl"):
    with open(file_path, "wb") as f:
        pickle.dump(vocab, f)
    print(f"✅ Saved vocab to {file_path}")


# In[10]:


def load_vocab(file_path="coco_vocab.pkl"):
    with open(file_path, "rb") as f:
        vocab = pickle.load(f)
    print(f"✅ Loaded vocab from {file_path}")
    return vocab


# In[11]:


import os

def get_or_build_vocab(annotations_path, vocab_path="coco_vocab.pkl", threshold=10):
    if os.path.exists(vocab_path):
        return load_vocab(vocab_path)
    else:
        vocab = build_vocab(annotations_path, threshold=threshold)
        save_vocab(vocab, vocab_path)
        return vocab


# In[12]:


annotations_file = "/home/pdas4/vitcap_r/data/MSCOCO/annotations/captions_train2014.json"
vocab_file = "coco_vocab.pkl"

vocab = get_or_build_vocab(annotations_file, vocab_path=vocab_file)


# In[13]:


#Training the PatchAttentionDecoder


# In[21]:


from transformers import ViTModel
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


# In[26]:


import torch
import torch.nn as nn
from tqdm import tqdm

def train_patch_attention_decoder(model, dataloader, vocab, device, num_epochs=5, start_epoch=1, lr=1e-4):
    model.to(device)
    criterion = nn.CrossEntropyLoss(ignore_index=vocab['<pad>'])
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    model.train()
    best_loss = float("inf")
    for epoch in range(start_epoch, start_epoch + num_epochs):
        total_loss = 0.0
        progress = tqdm(dataloader, desc=f"Epoch {epoch}/{start_epoch + num_epochs - 1}")
        for images, captions, _ in progress:
            images = torch.stack(images).to(device)            # (B, 3, 224, 224)
            captions = torch.stack(captions).to(device)        # (B, T)

            outputs, _ = model(images, captions)               # outputs: (B, T, vocab_size)
            targets = captions[:, 1:]                          # remove <startseq>
            outputs = outputs[:, :-1, :]                       # align with targets

            if outputs.size(1) != targets.size(1):
                min_len = min(outputs.size(1), targets.size(1))
                outputs = outputs[:, :min_len, :]
                targets = targets[:, :min_len]

            # 🔍 Debug print
            #print("Output shape:", outputs.shape)  # [batch_size, seq_len, vocab_size]
            #print("Target shape:", targets.shape)  # [batch_size, seq_len]
            loss = criterion(outputs.reshape(-1, outputs.size(-1)), targets.reshape(-1))
            total_loss += loss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            progress.set_postfix(loss=loss.item())

       #print(f"✅ Epoch {epoch+1} - Avg Loss: {total_loss / len(dataloader):.4f}")
        avg_loss = total_loss / len(dataloader)
        print(f"✅ Epoch {epoch+1} - Avg Loss: {avg_loss:.4f}")
       
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), "/home/pdas4/vitcap_r/models/patch_attention_best.pth")
            print(f"📦 Saved new best model (loss: {avg_loss:.4f})")




# In[27]:


# Load vocab
vocab = load_vocab("coco_vocab.pkl")

# Init model
model = PatchAttentionDecoder(embed_dim=256, hidden_dim=256, vocab_size=len(vocab))

# 💾 Load best checkpoint
#model.load_state_dict(torch.load("/home/pdas4/vitcap_r/models/patch_attention_best.pth"))
checkpoint = torch.load("/home/pdas4/vitcap_r/models/patch_attention_best.pth")
new_checkpoint = {k.replace('module.', ''): v for k, v in checkpoint.items()}
model.load_state_dict(new_checkpoint)


# Data Parallel across 2 GPUs
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = nn.DataParallel(model)
model.to(device)


# Prepare dataloader
train_loader = get_loader(
    image_dir="/home/pdas4/vitcap_r/data/MSCOCO/train2014",
    annotation_file="/home/pdas4/vitcap_r/data/MSCOCO/annotations/captions_train2014.json",
    vocab=vocab,
   # batch_size=32,
    batch_size=64,
    max_len=35
)

# Start training
#train_patch_attention_decoder(model, train_loader, vocab, device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
train_patch_attention_decoder(model, train_loader, vocab, device=device, num_epochs=24, start_epoch=97)

# In[ ]:




