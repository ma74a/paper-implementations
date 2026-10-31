
from torch.utils.data import DataLoader

from typing import Tuple
from omegaconf import OmegaConf

from .transforms import TrainTransform, ValTransform
from .dataset import VOC2012Segmentation

def get_data(cfg: OmegaConf) -> Tuple[DataLoader, DataLoader]:
    train_transform = TrainTransform()
    val_transform = ValTransform()
    
    train_dataset = VOC2012Segmentation(
        data_dir=cfg["data"]["root"],
        split="train",
        transforms=train_transform
    )
    val_dataset = VOC2012Segmentation(
        data_dir=cfg["data"]["root"],
        split="val",
        transforms=val_transform
    )
    
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=16,
        shuffle=True
    )
    val_loader = DataLoader(
        dataset=val_dataset,
        batch_size=16,
        shuffle=False
    )
    
    return train_loader, val_loader