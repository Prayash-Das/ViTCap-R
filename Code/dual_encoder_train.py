'''import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from dual_encoder_model import DualEncoder, contrastive_loss
from utils import get_loader, load_vocab

def save_model(model, path):
    torch.save(model.state_dict(), path)
    print(f"💾 Model saved to: {path}")


def train_dual_encoder(model, dataloader, device, vocab, num_epochs=50, lr=1e-4):
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    best_loss = float('inf')

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        loop = tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}")

        for images, captions, _ in loop:
            images = torch.stack(images).to(device)
            captions = torch.stack(captions).to(device)

            img_vecs, txt_vecs = model(images, captions)
            loss = contrastive_loss(img_vecs, txt_vecs)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            loop.set_postfix(loss=loss.item())

        avg_loss = total_loss / len(dataloader)
        print(f"✅ Epoch {epoch+1} - Avg Contrastive Loss: {avg_loss:.4f}")

        # Save model if best
        if avg_loss < best_loss:
            best_loss = avg_loss
            save_model(model, "/home/pdas4/vitcap_r/models/dual_encoder_best.pth")

# -------------------------
# Main Entry
# -------------------------
if __name__ == "__main__":
    vocab = load_vocab("/home/pdas4/vitcap_r/data/coco_vocab.pkl")
    train_loader = get_loader(
        image_dir="/home/pdas4/vitcap_r/data/MSCOCO/train2014",
        annotation_file="/home/pdas4/vitcap_r/data/MSCOCO/annotations/captions_train2014.json",
        vocab=vocab,
        batch_size=32,
        max_len=35
    )

    model = DualEncoder(embed_dim=256, vocab_size=len(vocab), hidden_dim=256)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.load_state_dict(torch.load("/home/pdas4/vitcap_r/models/dual_encoder_best.pth", map_location=device))
    model.to(device)
    train_dual_encoder(model, train_loader, device, vocab)'''

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from dual_encoder_model import DualEncoder, contrastive_loss
from utils import get_loader, load_vocab

def save_model(model, path):
    torch.save(model.state_dict(), path)
    print(f"💾 Model saved to: {path}")

def train_dual_encoder(model, dataloader, device, vocab, num_epochs=62, start_epoch=59, lr=1e-4):
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    best_loss = float('inf')

    for epoch in range(start_epoch, start_epoch + num_epochs):
        model.train()
        total_loss = 0
        loop = tqdm(dataloader, desc=f"Epoch {epoch}/{start_epoch + num_epochs - 1}")

        for images, captions, _ in loop:
            images = torch.stack(images).to(device)
            captions = torch.stack(captions).to(device)

            img_vecs, txt_vecs = model(images, captions)
            loss = contrastive_loss(img_vecs, txt_vecs)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            loop.set_postfix(loss=loss.item())

        avg_loss = total_loss / len(dataloader)
        print(f"✅ Epoch {epoch} - Avg Contrastive Loss: {avg_loss:.4f}")

        # Save model if best
        if avg_loss < best_loss:
            best_loss = avg_loss
            save_model(model, "/home/pdas4/vitcap_r/models/dual_encoder_best.pth")

# -------------------------
# Main Entry
# -------------------------
if __name__ == "__main__":
    vocab = load_vocab("/home/pdas4/vitcap_r/data/coco_vocab.pkl")
    train_loader = get_loader(
        image_dir="/home/pdas4/vitcap_r/data/MSCOCO/train2014",
        annotation_file="/home/pdas4/vitcap_r/data/MSCOCO/annotations/captions_train2014.json",
        vocab=vocab,
        batch_size=32,
        max_len=35
    )

    model = DualEncoder(embed_dim=256, vocab_size=len(vocab), hidden_dim=256)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.load_state_dict(torch.load("/home/pdas4/vitcap_r/models/dual_encoder_best.pth", map_location=device))
    model.to(device)

    # 🚀 Training 70 more epochs to reach total 120
    train_dual_encoder(model, train_loader, device, vocab, num_epochs=62, start_epoch=59)

