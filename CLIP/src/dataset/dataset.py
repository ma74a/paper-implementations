import torch
from torch.utils.data import Dataset
from torchvision import transforms as T
from transformers import CLIPTokenizer

from typing import List, Tuple
import pandas as pd
import os
from PIL import Image
import random

class ImageTextDataset(Dataset):
    def __init__(
        self,
        images_dir: str,
        caption_path: str,
        image_names: List[str],
        tokenizer: CLIPTokenizer,
        transforms: T,
        train: bool=True
    ) -> None:
        self.images_dir = images_dir
        self.tokenizer = tokenizer
        self.transforms = transforms
        self.train = train
        
        # read the caption file as DataFrame [image, caption]
        df = pd.read_csv(caption_path)
        # filter the df to get only the [images, caption]
        # that in images_names parameter
        df = df[df["image"].isin(set(image_names))]
        
        # {"img1.jpg": [cap1, cap2, cap3, cap4, cap5], ...}
        # for each image get its all 5 catpion to get randomly one of then in train
        self.captions = df.groupby("image")["caption"].apply(list).to_dict()
        # get the image
        self.images = sorted(image_names)
        
    def __len__(self) -> int:
        return len(self.images)
    
    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        image_name = self.images[index]
        image_path = os.path.join(self.images_dir, image_name)
        
        captions = self.captions[image_name]
        if self.train:
            caption = random.choice(captions)
        else:
            caption = captions[0]
            
        img = Image.open(image_path).convert("RGB")
        if self.transforms:
            img = self.transforms(img)
            
        # give us dict{input_ids: ..., attention_mask: ...}
        # we need input_ids but it is in shape -> [1, 77]
        # so we squeeze it in first dim to remove it 
        encoded_text = self.tokenizer(
            caption,
            padding="max_length",
            truncation=True,
            max_length=77,
            return_tensors="pt",
        )  
        encoded_text = encoded_text["input_ids"].squeeze(0)
        
        return img, encoded_text
        
        