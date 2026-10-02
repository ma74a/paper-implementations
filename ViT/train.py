import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import LambdaLR
from torchvision.transforms import v2
from torchinfo import summary

from omegaconf import DictConfig, OmegaConf
import hydra
from hydra.core.hydra_config import HydraConfig
import logging
import os
import matplotlib.pyplot as plt
import json
import math
import pandas as pd
from tqdm import tqdm

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
    
    
    # get accuracy
    def accuracy_fn(y_pred, y_true):
        # y_pred shape -> [B, num_classes] this row logits from model
        # y_true shape -> [num_classes] this ground truth class indices
        # get the index of higher probs accros num_classes dim
        preds = torch.argmax(y_pred, dim=1)
        # compare preds and y_true and check how many matches
        correct = (preds == y_true).sum().item()
        
        # divide by the number of samples here
        acc = correct / y_true.size(0)
        
        return acc
    
    def lr_lambda(current_step: int):
        """
        Return the learning-rate multiplier for a linear-warmup and cosine-decay schedule.

        This function is designed to be used with PyTorch's ``LambdaLR`` scheduler.
        The returned value is multiplied by the optimizer's base learning rate.

        The schedule has two phases:

        1. Linear warmup:
        The learning-rate multiplier increases linearly from 0.0 to 1.0.

        2. Cosine decay:
        The multiplier decreases from 1.0 to 0.0 following a half-cosine curve.

        Args:
            current_step (int):
                Current optimizer step, not the current epoch.

        Returns:
            float:
                Learning-rate multiplier in the range [0.0, 1.0].

        Notes:
            ``num_warmup_steps`` and ``total_training_steps`` are defined in the
            enclosing scope.
        """
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        
        progress = float(current_step - num_warmup_steps) / float(max(1, total_training_steps - num_warmup_steps))
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))
    
    # Calculate the total number of training steps:
    # one step is performed for each batch, so:
    # total steps = number of epochs × number of batches per epoch
    total_training_steps = cfg.train.epochs * len(train_loader)
    # Use the first 23% of the total training steps as a warmup period.
    # During warmup, the learning rate gradually increases to the target learning rate.
    num_warmup_steps = int(0.23 * total_training_steps)
    
    optimizer = torch.optim.AdamW(
        vit_model.parameters(),
        lr=cfg.optimizer.lr,
        betas=(cfg.optimizer.beta1, cfg.optimizer.beta2),
        weight_decay=cfg.optimizer.weight_decay
    )
    criterion = nn.CrossEntropyLoss()
    
    # controls how the learning rate changes over time.
    scheduler = LambdaLR(optimizer=optimizer, lr_lambda=lr_lambda)
    
    # training loop will be below this
    log.info("MODEL TRAINING BEGINS...")
    train_acc = []
    train_loss = []
    learning_rates = []
    val_acc = []
    val_loss = []
    n_trains = len(train_loader)
    n_vals = len(val_loader)
    best_val_acc = 0
    vit_model = vit_model.to(device=device)
    for epoch in range(cfg.train.epochs):
        vit_model.train()
        train_loss_average = 0
        train_accuracy_average = 0
        for image, label in tqdm(train_loader):
            image, label = image.to(device), label.to(device)
            optimizer.zero_grad()
            learning_rates.append(scheduler.get_last_lr()[0])
            
            pred_logits = vit_model(image)
            loss = criterion(pred_logits, label)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(vit_model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            
            train_loss_average += loss.item()
            train_accuracy_average += accuracy_fn(y_pred=pred_logits, y_true=label)
            
        epoch_train_avg_loss = train_loss_average / n_trains
        epoch_train_avg_acc = train_accuracy_average / n_trains
        train_loss.append(epoch_train_avg_loss)
        train_acc.append(epoch_train_avg_acc)
        
        # eval_code
        val_loss_average = 0
        val_accuracy_average = 0
        vit_model.eval()
        with torch.inference_mode():
            for image, label in tqdm(val_loader):
                image, label = image.to(device), label.to(device)
                pred_logits = vit_model(image)
                loss = criterion(pred_logits, label)
                
                val_loss_average += loss.item()
                val_accuracy_average += accuracy_fn(y_pred=pred_logits, y_true=label)
                
            epoch_val_avg_loss = val_loss_average / n_vals
            epoch_val_avg_acc = val_accuracy_average / n_vals
            val_loss.append(epoch_val_avg_loss)
            val_acc.append(epoch_val_avg_acc)
            
            print(
                f"Epoch {epoch+1} | "
                f"train_loss: {epoch_train_avg_loss:.4f} | train_accuracy: {epoch_train_avg_acc:.4f} | "
                f"val_loss: {epoch_val_avg_loss:.4f} | val_accuracy: {epoch_val_avg_acc:.4f} | "
                f"LR: {scheduler.get_last_lr()[0]:.6f}"
            )
            
            log.info(f"Epoch {epoch+1} | "
            f"train_loss: {epoch_train_avg_loss:.4f} | train_accuracy: {epoch_train_avg_acc:.4f} | "
            f"val_loss: {epoch_val_avg_loss:.4f} | val_accuracy: {epoch_val_avg_acc:.4f} | "
            f"LR: {scheduler.get_last_lr()[0]:.6f}")
            
            # storing model for inference at later point of time===== model checkpointing ====
            if epoch_val_avg_acc > best_val_acc:
                best_val_acc = epoch_val_avg_acc
                model_path = os.path.join(output_dir, "best_model.pt")
                torch.save(vit_model.state_dict(), model_path)
                log.info(f"New best model saved to {model_path} , with accuracy : {best_val_acc}")
                
                
    learning_rates.append(scheduler.get_last_lr()[0])

    log.info("Storing the training artifacts detials")

    data = list(zip(train_loss,train_acc,val_loss,val_acc))
    df = pd.DataFrame(data,columns=['train loss','train acc','val loss','val acc'])
    
    csv_path = os.path.join(output_dir, "training_history.csv")
    df.to_csv(csv_path,index_label="epoch")
    log.info("WOrking on Training Plots...")
    saving_training_plots(df,learning_rates,output_dir)
    df = {'learning_rates_per_step':learning_rates}
    df = pd.DataFrame(df)
    csv_path = os.path.join(output_dir, "learning_rates_per_step.csv")
    df.to_csv(csv_path,index_label="steps")

    log.info(f"😎 Training Completed, details stored in {output_dir}")
        
    
    
    
if __name__ == "__main__":
    main()