import torch
from torch.utils.data import Dataset
from torchvision import transforms as T

import os
from PIL import Image


class VOC2012Segmentation(Dataset):
    def __init__(
        self,
        data_dir: str,
        split: str="train",
        transforms: T=None
    ) -> None:
        self.data_dir = data_dir
        self.split = split
        self.image_dir = os.listdir(os.path.join(data_dir, split, "images"))
        self.mask_dir = os.listdir(os.path.join(data_dir, split, "masks"))

        self.transforms = transforms
        
    def __len__(self) -> int:
        return len(self.image_dir)
    
    def __getitem__(self, idx: int):
        img_name = self.image_dir[idx]
        img_path = os.path.join(self.data_dir, self.split, "images", img_name)
        mask_name = self.mask_dir[idx]
        mask_path = os.path.join(self.data_dir, self.split, "masks", mask_name)
        
        img = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path)
        
        if self.transforms:
            img, mask = self.transforms(img, mask)
        
        return img, mask
        
# if __name__ == "__main__":
#     obj = VOC2012Segmentation(data_dir="/home/etman/Downloads/voc_subset")
#     img, mask = obj[0]
#     print(img.shape)