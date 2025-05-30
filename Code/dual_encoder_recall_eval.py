import torch
from tqdm import tqdm
from utils import get_loader, load_vocab
from dual_encoder_model_hnm import DualEncoder, contrastive_loss

def compute_recall_at_k(img_embeds, txt_embeds, k=[1,5,10]):
    recalls = {'i2t': {}, 't2i': {}}

    # Image-to-Text Retrieval
    sim_matrix = torch.matmul(img_embeds, txt_embeds.T)
    sorted_indices = torch.argsort(sim_matrix, dim=1, descending=True)
    
    for kk in k:
        correct = 0
        for i in range(img_embeds.size(0)):
            if i in sorted_indices[i, :kk]:
                correct += 1
        recalls['i2t'][f'Recall@{kk}'] = correct / img_embeds.size(0)

    # Text-to-Image Retrieval
    sim_matrix = torch.matmul(txt_embeds, img_embeds.T)
    sorted_indices = torch.argsort(sim_matrix, dim=1, descending=True)

    for kk in k:
        correct = 0
        for i in range(txt_embeds.size(0)):
            if i in sorted_indices[i, :kk]:
                correct += 1
        recalls['t2i'][f'Recall@{kk}'] = correct / txt_embeds.size(0)

    return recalls


def extract_features(model, dataloader, device):
    model.eval()
    img_embeds = []
    txt_embeds = []

    with torch.no_grad():
        for images, captions, _ in tqdm(dataloader, desc="Extracting Features"):
            images = torch.stack(images).to(device)
            captions = torch.stack(captions).to(device)

            img_vecs = model.forward_image(images)
            txt_vecs = model.forward_text(captions)

            img_embeds.append(img_vecs)
            txt_embeds.append(txt_vecs)

    img_embeds = torch.cat(img_embeds, dim=0)
    txt_embeds = torch.cat(txt_embeds, dim=0)

    return img_embeds, txt_embeds


def evaluate_recall_at_k(model_path, vocab_path, image_dir, annotation_file, device):
    # Load model
    vocab = load_vocab(vocab_path)
    model = DualEncoder(embed_dim=256, vocab_size=len(vocab), hidden_dim=256)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)

    # Prepare DataLoader (validation set)
    val_loader = get_loader(image_dir, annotation_file, vocab, batch_size=32, max_len=35)

    # Extract embeddings
    img_embeds, txt_embeds = extract_features(model, val_loader, device)
    img_embeds = img_embeds.cpu()
    txt_embeds = txt_embeds.cpu()
    # Compute Recall@K
    recalls = compute_recall_at_k(img_embeds, txt_embeds)

    print("🎯 Recall@K Evaluation:")
    for task in recalls:
        print(f"  {task.upper()}:")
        for k, v in recalls[task].items():
            print(f"    {k}: {v:.4f}")


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

evaluate_recall_at_k(
    model_path="/home/pdas4/vitcap_r/models/dual_encoder_hnm_best.pth",
    vocab_path="/home/pdas4/vitcap_r/data/coco_vocab.pkl",
    image_dir="/home/pdas4/vitcap_r/data/MSCOCO/val2014",
    annotation_file="/home/pdas4/vitcap_r/data/MSCOCO/annotations/captions_val2014.json",
    device=device
)
