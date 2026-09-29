import torch
from torch.utils.data import DataLoader
from torchvision.transforms import v2

from omegaconf import DictConfig, OmegaConf
import logging
import os
import matplotlib.pyplot as plt

from src import VisionTransformer,TomAndJerryDataset

if torch.cuda.is_available():
    device = torch.device("cuda")
    print("✅ Setting Device as CUDA...")
elif torch.backends.mps.is_available() and torch.backends.mps.is_built():
    print("🫡  Device is set to MPS...")
    device = torch.device('mps')
else:
    print("No accelerator available 🥺 ...using CPU for this task...")
    device = torch.device("cpu")
    
log = logging.getLogger(__name__)

train_transform = v2.Compose([
    v2.RandomResizedCrop(size=(224, 224), scale=(0.8, 1.0)), 
    v2.TrivialAugmentWide(),   
    v2.RandomHorizontalFlip(p=0.5),
    v2.ToTensor(),
    v2.Normalize(mean=[0.485, 0.456, 0.406],
                 std=[0.229, 0.224, 0.225])
])

val_transform = v2.Compose([
    v2.Resize((224, 224)),
    v2.ToTensor(),
    v2.Normalize(mean=[0.485, 0.456, 0.406],
                 std=[0.229, 0.224, 0.225])
])


def load_data(cfg: DictConfig):
    train_path = cfg.train_dataset_path
    val_path = cfg.val_dataset_path
    
    train_dataset = TomAndJerryDataset(
        data_dir=train_path,
        transforms=train_transform
    )
    val_dataset = TomAndJerryDataset(
        data_dir=val_path,
        transforms=val_transform
    )

    train_loader = DataLoader(train_dataset,
                                   batch_size=cfg.batch_size,
                                   shuffle=True)
    val_loader = DataLoader(val_dataset,
                                 batch_size=cfg.batch_size,
                                 shuffle=False)

    return train_dataset, val_dataset, train_loader, val_loader


def saving_training_plots(history_df, lr, output_dir):
    """
    Function to store train vs. val loss and train vs. val accuracy.

    Input : history df with columns 
                'train loss','train acc','val loss','val acc','learning rate'
    Output : NONe
        
    """
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(1,3, figsize=(15,5))
    ax[0].plot(history_df["train loss"], label="Train Loss")
    ax[0].plot(history_df["val loss"], label="Validation Loss")
    ax[0].set_title("Loss Curves")
    ax[0].set_xlabel("Epoch")
    ax[0].legend()

    ax[1].plot(history_df["train acc"], label="Train Accuracy")
    ax[1].plot(history_df["val acc"], label="Validation Accuracy")
    ax[1].set_title("Accuracy Curves")
    ax[1].set_xlabel("Epoch")
    ax[1].legend()

    ax[2].plot(lr, label="Learning rate per step")
    ax[2].set_title("Learning Rate Curve")
    ax[2].set_xlabel("steps")
    ax[2].legend()

    plot_path = os.path.join(output_dir, "training_curves.png")
    fig.savefig(plot_path)
    log.info(f"Saved training plot to {plot_path}")