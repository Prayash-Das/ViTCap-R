import os
import pickle
from pycocotools.coco import COCO
from nltk.tokenize import TreebankWordTokenizer
from collections import Counter
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

# Load vocab from pickle
def load_vocab(file_path):
    with open(file_path, "rb") as f:
        vocab = pickle.load(f)
    return vocab

# COCO Dataset
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
        self.tokenizer = TreebankWordTokenizer()

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

        tokens = ['startseq'] + self.tokenizer.tokenize(caption.lower()) + ['endseq']
        caption_ids = [self.vocab.get(tok, self.vocab['<unk>']) for tok in tokens]
        caption_ids = caption_ids[:self.max_len] + [self.vocab['<pad>']] * (self.max_len - len(caption_ids))

        return image, torch.tensor(caption_ids), img_id

# Dataloader
def get_loader(image_dir, annotation_file, vocab, batch_size=32, max_len=35):
    dataset = CocoDataset(image_dir, annotation_file, vocab, max_len=max_len)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=lambda x: tuple(zip(*x)))
