import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from torch.utils.data import DataLoader

from src.dataset import CustomDataset
from utils.config import Config

def load_data():
    train_dataset = CustomDataset(data_dir=Config.TRAIN_DATA, transforms=Config.TRANSFORMS["train"])
    val_dataset = CustomDataset(data_dir=Config.VAL_DATA, transforms=Config.TRANSFORMS["val"])

    class_to_idx = train_dataset.class_to_idx

    train_loader = DataLoader(dataset=train_dataset, batch_size=Config.BATCH_SIZE, shuffle=True, num_workers=4)
    val_loader = DataLoader(dataset=val_dataset, batch_size=16, shuffle=False, num_workers=3)

    return train_loader, val_loader