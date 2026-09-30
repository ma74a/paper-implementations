import torch
from torchinfo import summary
from torch.utils.data import DataLoader
from torchvision.transforms import v2

from omegaconf import DictConfig, OmegaConf
import hydra
from hydra.core.hydra_config import HydraConfig
import logging
import os
import matplotlib.pyplot as plt
import json

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
    train_path = cfg.train.train_dataset_path
    val_path = cfg.train.val_dataset_path
    
    train_dataset = TomAndJerryDataset(
        data_dir=train_path,
        transforms=train_transform
    )
    val_dataset = TomAndJerryDataset(
        data_dir=val_path,
        transforms=val_transform
    )

    train_loader = DataLoader(train_dataset,
                                   batch_size=cfg.train.batch_size,
                                   shuffle=True)
    val_loader = DataLoader(val_dataset,
                                 batch_size=cfg.train.batch_size,
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
    

@hydra.main(config_path="configs", config_name="config", version_base=None)
def main(cfg: DictConfig):
    # print and log the active config and the Hydra output directory:
    print(f"Current working directory: {os.getcwd()}")
    output_dir = HydraConfig.get().runtime.output_dir
    log.info(f"All artifacts will be saved in {output_dir}")
    log.info(f"\n{OmegaConf.to_yaml(cfg)}")
    
    log.info("Dataset creation begin")
    train_dataset, val_dataset, train_loader, val_loader = load_data(cfg=cfg)
    num_classes = len(train_dataset.classes)
    log.info("Dataset created")
    
    log.info(f"Dataset Classes and Corresponding Labels : {train_dataset.class_to_idx}")

    try:
        log.info("Verifying consistency between config and dataset...")
        assert num_classes == cfg.model.num_classes, \
            f"Mismatch: config expects {cfg.model.num_classes} classes, but dataset has {num_classes}."
        log.info("✅ Verification successful.")

    except AssertionError as e:
        log.error(f"CONFIGURATION ERROR: {e}")
        import sys
        sys.exit(1)
        
    log.info("Storing the index vs label mapping for the Current Dataset")
    idx_to_class = train_dataset.idx_to_class
    mapping_save_path = os.path.join(output_dir, "mapping_saved_file.json")
    with open(mapping_save_path, 'w+') as f:
        json.dump(idx_to_class, f, indent=4)
    log.info(f"Mapping saved at : {mapping_save_path}")
    
    log.info("Model Creation Begin")
    vit_model = VisionTransformer(
        num_classes=cfg.model.num_classes,
        input_channel=cfg.model.in_channels,
        image_size=cfg.model.img_size,
        patch_size=cfg.model.patch_size,
        embedding_dim=cfg.model.embed_dim,
        input_dropout_rate=cfg.model.input_dropout_rate,
        num_encoder_blocks=cfg.model.num_of_encoders,
        num_heads=cfg.model.num_heads,
        dff_scale=cfg.model.dff_scale_factor,
        attention_dropout_rate=cfg.model.attention_dropout_rate,
        ff_dropout_rate=cfg.model.ff_dropout_rate
    )
    test_input = torch.randn(
        cfg.train.batch_size,
        cfg.model.in_channels,
        cfg.model.img_size,
        cfg.model.img_size
    )
    # print(summary(vit_model, input_data=test_input))
    log.info("--- Model Summary ---")
    log.info(summary(vit_model, input_data=test_input))
    log.info("--------------------")
    
    
if __name__ == "__main__":
    main()