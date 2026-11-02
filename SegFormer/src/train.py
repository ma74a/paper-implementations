import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))

import torch
from torch import optim
from torchinfo import summary


import os
import math
from tqdm import tqdm
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

from dataset import get_data
from model import SegFormer
from utils import load_config, setup_logger, create_experiment_dir





if torch.cuda.is_available():
    device = torch.device("cuda")
    print("✅ Setting Device as CUDA...")
elif torch.backends.mps.is_available() and torch.backends.mps.is_built():
    print("🫡  Device is set to MPS...")
    device = torch.device('mps')
else:
    print("No accelerator available 🥺 ...using CPU for this task...")
    device = torch.device("cpu")
    
    
# @hydra.main(config_path="../configs", config_name="config", version_base=None)
def main():
    config_path = "/home/etman/etman/paper-implementations/SegFormer/configs/config.yaml"
    config = load_config(config_path=config_path)
    # print and log the active config and the Hydra output directory:
    print(f"Current working directory: {os.getcwd()}")
     # Create experiment directory
    experiment_dir = create_experiment_dir(config_path)
    output_dir = os.path.join(os.getcwd(), "outputs")
    logger = setup_logger(experiment_dir)
   
    
    logger.info("Dataset creation begin")
    train_loader, val_loader = get_data(cfg=config)
    logger.info("Dataset created")
    
    logger.info("Model Creation Begin")
    segformer_model = SegFormer(num_classes=21)
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
    
    logger.info("Start Training")
    train_losses = []
    val_losses = []
    learning_rates = []
    n_trains = len(train_loader)
    n_vals = len(val_loader)
    best_val_loss = float("inf")
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
        
        val_avg_loss = 0
        segformer_model.eval()
        with torch.inference_mode():
            for image, mask in tqdm(val_loader, desc="Validating"):
                image, mask = image.to(device), mask.to(device)
                
                output = segformer_model(image)
                loss = criterion(output, mask)
                
                val_avg_loss += loss.item()
                
            epoch_val_avg_loss = val_avg_loss / n_vals
            val_losses.append(epoch_val_avg_loss)
            
            print(f"Epoch: {epoch + 1} | train_loss: {epoch_train_avg_loss} | val_loss: {epoch_val_avg_loss} | learning_rate: {scheduler.get_last_lr()[0]:0.6f}")
     
    
    
    
    
if __name__ == "__main__":
    main()
    