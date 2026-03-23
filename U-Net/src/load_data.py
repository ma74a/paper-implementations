import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from torch.utils.data import DataLoader, random_split

from .dataset import SegmentationDataset
from utils.config import Config

def load_data():
    dataset = SegmentationDataset(
        images_path=Config.IMAGES,
        masks_path=Config.MASKS,
        img_transforms=Config.IMAGE_TRANSFORMS,
        mask_transforms=Config.MASK_TRANSFORMS
    )
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    # print(len(train_dataset), len(val_dataset))
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)

    return train_loader, val_loader