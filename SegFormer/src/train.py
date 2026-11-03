import sys
from pathlib import Path
# Add project root to Python path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))
# print(PROJECT_ROOT)


import torch
from torch import optim
from torchinfo import summary


import os
import math
from tqdm import tqdm
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
import csv

from dataset import get_data
from model import SegFormer
from utils import (
    load_config,
    setup_logger,
    create_experiment_dir,
    MeanIoU,
    plot_confusion_matrix,
    plot_training_history
)





if torch.cuda.is_available():
    device = torch.device("cuda")
    print("✅ Setting Device as CUDA...")
elif torch.backends.mps.is_available() and torch.backends.mps.is_built():
    print("🫡  Device is set to MPS...")
    device = torch.device('mps')
else:
    print("No accelerator available 🥺 ...using CPU for this task...")
    device = torch.device("cpu")
    

def save_checkpoint(
    model,
    optimizer,
    scheduler,
    epoch,
    miou,
    save_path,
):
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "miou": miou,
    }

    torch.save(checkpoint, save_path)

# @hydra.main(config_path="../configs", config_name="config", version_base=None)
def main():
    config_path = (PROJECT_ROOT / "configs" / "config.yaml")
   
    config = load_config(config_path=str(config_path))
    # print and log the active config and the Hydra output directory:
    print(f"Current working directory: {os.getcwd()}")
     # Create experiment directory
    experiment_dir = create_experiment_dir(str(config_path))
    experiment_dir.mkdir(parents=True,exist_ok=True)
    checkpoint_dir = experiment_dir / "checkpoints"
    confusion_dir = experiment_dir / "confusion_matrices"
    plots_dir = experiment_dir / "plots"
    for directory in [
        checkpoint_dir,
        confusion_dir,
        plots_dir,
    ]:
        directory.mkdir(
            parents=True,
            exist_ok=True,
        )
    # output_dir = os.path.join(os.getcwd(), "outputs")
    # logger = setup_logger(experiment_dir)
    logger = setup_logger(str(experiment_dir))

    logger.info(f"Device: {device}")
    logger.info(f"Experiment directory: {experiment_dir}")
   
    
    logger.info("Dataset creation begin")
    train_loader, val_loader = get_data(cfg=config)
    logger.info("Dataset created")
    
    logger.info(f"Training batches: {len(train_loader)}")
    logger.info(f"Validation batches: {len(val_loader)}")
    
    logger.info("Model Creation Begin")
    class_names = config["data"]["class_names"]
    num_classes = len(class_names)
    segformer_model = SegFormer(num_classes=num_classes).to(device)
    batch_size = config["train"]["batch_size"]
    test_images = torch.randn(
        batch_size,
        config["data"]["in_channels"],
        config["data"]["img_size"],
        config["data"]["img_size"],
        device=device
    )
    logger.info("--- SegFormer Model Summary ---")
    logger.info(
        summary(
            segformer_model,
            input_data=(test_images),
            device=device,
            verbose=1,
        )
    )
    logger.info("--------------------------")
    del test_images
    
    power = config["train"].get("power", 0.9)
    warmup_start_factor = config["train"].get("warmup_start_factor", 0.1)

    iters_per_epoch = len(train_loader)
    max_iters = config["train"]["epochs"] * iters_per_epoch
    warmup_iters = int(0.01 * max_iters)
    
    
    def lr_lambda(t):
        poly = (1 - t / max_iters) ** power
        if t < warmup_iters:
            alpha = t / warmup_iters
            return poly * (warmup_start_factor + (1 - warmup_start_factor) * alpha)
        return poly
    
    optimizer = optim.AdamW(
        segformer_model.parameters(),
        lr=config["train"]["lr"],
        weight_decay=config["train"]["weight_decay"]
    )
    scheduler = optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lr_lambda
    )
    criterion = torch.nn.CrossEntropyLoss(ignore_index=255)
    miou_metric = MeanIoU(
        num_classes=num_classes,
        ignore_index=255,
        device=device,
    )
    
    logger.info("Start Training")
    train_losses = []
    val_losses = []
    val_mious = []
    learning_rates = []
    epoch_lrs = []   # one LR per epoch, for metrics.csv
    n_trains = len(train_loader)
    n_vals = len(val_loader)
    best_confusion_matrix = None
    best_miou = float("-inf")
    for epoch in range(config["train"]["epochs"]):
        train_avg_loss = 0
        segformer_model.train()
        for image, mask in tqdm(train_loader, desc="Training"):
            image, mask = image.to(device), mask.to(device)
            
            optimizer.zero_grad()
            learning_rates.append(scheduler.get_last_lr()[0])
            output = segformer_model(image)
            loss = criterion(output, mask)
            loss.backward()
            optimizer.step()
            scheduler.step()
            
            train_avg_loss += loss.item()
            
        epoch_train_avg_loss = train_avg_loss / n_trains
        train_losses.append(epoch_train_avg_loss)
        
        # Reset metric once before validation
        miou_metric.reset()
        val_avg_loss = 0
        segformer_model.eval()
        with torch.inference_mode():
            for image, mask in tqdm(val_loader, desc="Validating"):
                image, mask = image.to(device), mask.to(device)
                
                output = segformer_model(image)
                loss = criterion(output, mask)
                
                val_avg_loss += loss.item()
                # Accumulate confusion matrix
                miou_metric.update(output, mask)
                
            epoch_val_avg_loss = val_avg_loss / n_vals
            # Compute dataset-level metrics
            metrics = miou_metric.compute()
            val_losses.append(epoch_val_avg_loss)
            
            epoch_miou = float(metrics["miou"])
            per_class_iou = metrics["per_class_iou"]
            confusion_matrix = metrics["confusion_matrix"]
            val_mious.append(epoch_miou)
            epoch_lrs.append(scheduler.get_last_lr()[0])
            
            print(f"Epoch: {epoch + 1} | train_loss: {epoch_train_avg_loss} | val_loss: {epoch_val_avg_loss} | mIoU: {epoch_miou} | learning_rate: {scheduler.get_last_lr()[0]:0.6f}")
            
            logger.info(
            f"Epoch [{epoch + 1}] | "
            f"Train Loss: {epoch_train_avg_loss:.4f} | "
            f"Val Loss: {epoch_val_avg_loss:.4f} | "
            f"mIoU: {epoch_miou:.4f} | "
            f"LR: {scheduler.get_last_lr()[0]:.6f}"
        )

        for class_id, class_iou in enumerate(per_class_iou):

            logger.info(
                f"Class {class_id} "
                f"({class_names[class_id]}): "
                f"IoU = {class_iou.item():.4f}"
            )
            
        if epoch_miou > best_miou:
            best_miou = epoch_miou
            save_checkpoint(
                model=segformer_model,
                optimizer=optimizer,
                scheduler=scheduler,
                epoch=epoch + 1,
                miou=best_miou,
                save_path=checkpoint_dir / "best_model.pth",
            )
            best_confusion_matrix = confusion_matrix.detach().cpu().clone()

            logger.info(
                f"New best mIoU: {best_miou:.4f}"
            )
            
        history_path = experiment_dir / "metrics.csv"
        with open(history_path,"w",newline="") as csv_file:
                writer = csv.writer(csv_file)
                writer.writerow([
                    "epoch",
                    "train_loss",
                    "val_loss",
                    "val_miou",
                    "learning_rate",
                ])
                for idx in range(len(train_losses)):
                    writer.writerow([
                        idx + 1,
                        train_losses[idx],
                        val_losses[idx],
                        val_mious[idx],
                        learning_rates[idx],
                    ])
    plot_training_history(
        train_losses=train_losses,
        val_losses=val_losses,
        val_mious=val_mious,
        save_dir=plots_dir / "plots.png"
    )
    plot_confusion_matrix(
        confusion_matrix=best_confusion_matrix,
        class_names=class_names,
        save_path=confusion_dir / "best_confusion_matrix.png",
    )

    logger.info(
        f"Best validation mIoU: {best_miou:.4f}"
    )
    
    
    
    
if __name__ == "__main__":
    main()
    