from torch.utils.data import DataLoader
from torchvision import transforms as T
from transformers import CLIPTokenizer

from omegaconf import DictConfig
import pandas as pd
from typing import Tuple, List
import random

from .dataset import ImageTextDataset


def get_transforms(image_size: int=224, train: bool=True):
    # CLIP's normalization statistics
    normalize = T.Normalize(
        mean=(0.48145466, 0.4578275, 0.40821073),
        std=(0.26862954, 0.26130258, 0.27577711),
    )
    if train:
        return T.Compose([
            T.RandomResizedCrop(image_size, scale=(0.5, 1.0)),
            T.ToTensor(),
            normalize,
        ])
    return T.Compose([
        T.Resize(image_size),
        T.CenterCrop(image_size),
        T.ToTensor(),
        normalize,
    ])

def split_data(
    caption_path: str,
    val_ratio: float=0.15,
    seed: int=21
) -> Tuple[List[str], List[str]]:
    df = pd.read_csv(caption_path)
    image_names = sorted(df["image"].unique())
    
    rng = random.Random(seed)
    rng.shuffle(image_names)
    num_val = int(len(image_names) * val_ratio)
    # train_size, val_size = len(image_names) - num_val, num_val

    train = image_names[num_val:]
    val = image_names[:num_val]

    return train, val


def load_data(cfg: DictConfig, tokenizer: CLIPTokenizer) -> Tuple[DataLoader, DataLoader]:
    train_images, val_images = split_data("/home/etman/Downloads/flickr8k_1000/captions.txt")
    
    train_dataset = ImageTextDataset(
        images_dir=cfg["data"]["images_dir"],
        caption_path=cfg["data"]["captions_file"],
        image_names=train_images,
        tokenizer=tokenizer,
        transforms=get_transforms(train=True),
        train=True
    )
    val_dataset = ImageTextDataset(
        images_dir=cfg["data"]["images_dir"],
        caption_path=cfg["data"]["captions_file"],
        image_names=val_images,
        tokenizer=tokenizer,
        transforms=get_transforms(train=False),
        train=False
    )
    
    train_loader = DataLoader(
        train_dataset, batch_size=cfg["training"]["batch_size"], shuffle=True,
        num_workers=4, drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=cfg["training"]["batch_size"], shuffle=False,
        num_workers=4, drop_last=True,
    )
    # first_batch = next(iter(train_loader))
    
    # images, encoded_texts = first_batch
    # print(images.shape)
    # print(encoded_texts.shape)
    
    
    return train_loader, val_loader
