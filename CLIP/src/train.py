import torch
from torch import optim
from transformers import CLIPTokenizer
from torchinfo import summary

import logging
import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf
import os
import math
from tqdm import tqdm
import pandas as pd
import matplotlib.pyplot as plt

from model import CLIP
from loss import clip_contrastive_loss
from dataset import load_data
from visualize import show_batch, saving_training_plots

log = logging.getLogger(__name__)

if torch.cuda.is_available():
    device = torch.device("cuda")
    print("✅ Setting Device as CUDA...")
elif torch.backends.mps.is_available() and torch.backends.mps.is_built():
    print("🫡  Device is set to MPS...")
    device = torch.device('mps')
else:
    print("No accelerator available 🥺 ...using CPU for this task...")
    device = torch.device("cpu")




def get_params_group(
    model: CLIP,
    weight_decay: float=0.2
):
    """
    Paper: decoupled weight decay on all weights that are not gains or biases.
      gains  = LayerNorm weight (1-D)
      biases = every bias (1-D)
    The learnable temperature logit_scale is a scalar (0-D), so it is also left
    out of the decay.  Rule used: parameters with ndim < 2 get no decay.
    """
    
    decay_params = []
    no_decay_params = []
    
    for name, params in model.named_parameters():
        if not params.requires_grad:
            continue
        
        if params.ndim < 2: # biases, LayerNorm gains and biases, logit_scale
            no_decay_params.append(params)
        else: # linear / conv / embedding weights, cls token, position, attention
            decay_params.append(params)
            
    return [
            {"params": decay_params, "weight_decay": weight_decay},
            {"params": no_decay_params, "weight_decay": 0.0},
        ]

def lr_multiplier(step: int, warmup_steps: int, total_steps: int, min_ratio: float = 0.0) -> float:
    """Factor applied to the base lr: linear warmup, then cosine decay to min_ratio."""
    if step < warmup_steps:
        return (step + 1) / warmup_steps
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    return min_ratio + (1.0 - min_ratio) * 0.5 * (1.0 + math.cos(math.pi * progress))

@hydra.main(config_path="../configs", config_name="config", version_base=None)
def main(cfg: DictConfig):
    # print and log the active config and the Hydra output directory:
    print(f"Current working directory: {os.getcwd()}")
    output_dir = HydraConfig.get().runtime.output_dir
    log.info(f"All artifacts will be saved in {output_dir}")
    log.info(f"\n{OmegaConf.to_yaml(cfg)}")
    
    log.info("get the tokenizer")
    tokenizer = CLIPTokenizer.from_pretrained(
        cfg["tokenizer"]["name"]
    )
    
    
    log.info("DataLoader creation begin")
    train_loader, val_loader = load_data(cfg=cfg, tokenizer=tokenizer)
    log.info("DataLoader created")
    
    log.info("Model Creation Begin")
    clip_model = CLIP(
        embedding_dim=cfg["model"]["embed_dim"],
        image_size=cfg["data"]["image_size"],
        patch_size=cfg["model"]["image_encoder"]["patch_size"],
        in_channels=cfg["data"]["in_channels"],
        image_embedding_dim=cfg["model"]["image_encoder"]["image_embed"],
        num_heads=cfg["model"]["image_encoder"]["heads"],
        dff_scale=cfg["model"]["dff_scale"],
        num_encoder_blocks=cfg["model"]["image_encoder"]["layers"],
        dropout=cfg["model"]["dropout"],
        vocab_size=cfg["model"]["text_encoder"]["vocab_size"],
        context_length=cfg["tokenizer"]["context_length"],
        text_embedding_dim=cfg["model"]["text_encoder"]["text_embed"],
    )
    batch_size = cfg["training"]["batch_size"]
    test_images = torch.randn(
        batch_size,
        cfg["data"]["in_channels"],
        cfg["data"]["image_size"],
        cfg["data"]["image_size"],
        device=device
    )
    test_text = torch.randint(
        low=0,
        high=cfg["model"]["text_encoder"]["vocab_size"],
        size=(batch_size, cfg["tokenizer"]["context_length"]),
        device=device
    )
    log.info("--- CLIP Model Summary ---")
    log.info(
        summary(
            clip_model,
            input_data=(test_images, test_text),
            device=device,
            verbose=1,
        )
    )
    log.info("--------------------------")
    
    
    optimizer = optim.AdamW(
        get_params_group(model=clip_model, weight_decay=cfg["optimizer"]["weight_decay"]),
        lr=cfg["optimizer"]["lr"],
        betas=cfg["optimizer"]["betas"],
        eps=cfg["optimizer"]["eps"]
    )
    # len(train_loader) // batch_size = res with drop_last=True
    steps_per_epoch = len(train_loader)     
    # num of res above * num_epochs
    total_steps = cfg["training"]["epochs"] * steps_per_epoch 
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda step: lr_multiplier(
            step,
            warmup_steps=cfg["scheduler"]["warmup_steps"],
            total_steps=total_steps,
            min_ratio=cfg["scheduler"]["min_lr_ratio"],),
    )
    
    # training loop will be below this
    log.info("MODEL TRAINING BEGINS...")
    train_losses = []
    val_losses = []
    learning_rates = []
    n_trains = len(train_loader)
    n_vals = len(val_loader)
    best_val_loss = float("inf")
    for epoch in range(cfg["training"]["epochs"]):
        train_loss_avg = 0.0
        clip_model.train()
        for images, texts in tqdm(train_loader):
            images, texts = images.to(device), texts.to(device)
            
            optimizer.zero_grad()
            learning_rates.append(scheduler.get_last_lr()[0])
            
            images_logits, text_logits = clip_model(images, texts)
            loss = clip_contrastive_loss(images_logits, text_logits)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                clip_model.parameters(),
                cfg["training"]["grad_clip_norm"]
            )
            optimizer.step()
            scheduler.step()
            
            train_loss_avg += loss.item()
            
        epoch_train_avg_loss = train_loss_avg / n_trains
        train_losses.append(epoch_train_avg_loss)
        
        val_loss_avg = 0.0
        clip_model.eval()
        with torch.inference_mode():
            for images, texts in tqdm(val_loader):
                images, texts = images.to(device), texts.to(device) 
                
                images_logits, text_logits = clip_model(images, texts)
                loss = clip_contrastive_loss(images_logits, text_logits)
                
                val_loss_avg += loss.item()
                
            epoch_val_avg_loss = val_loss_avg / n_vals
            val_losses.append(epoch_val_avg_loss)
            
            print(
                f"\nEpoch {epoch+1} | "
                f"train_loss: {epoch_train_avg_loss:.4f} | "
                f"val_loss: {epoch_val_avg_loss:.4f} | "
                f"LR: {scheduler.get_last_lr()[0]:.6f}"
            )
            
            log.info(f"\nEpoch {epoch+1} | "
            f"train_loss: {epoch_train_avg_loss:.4f} | "
            f"val_loss: {epoch_val_avg_loss:.4f} | "
            f"LR: {scheduler.get_last_lr()[0]:.6f}")
            
            if best_val_loss > epoch_val_avg_loss:
                best_val_loss = epoch_val_avg_loss
                best_epoch = epoch
                model_path = os.path.join(output_dir, "best_model.pt")
                torch.save(clip_model.state_dict(), model_path)
                log.info(f"New best model saved to {model_path} , with loss : {best_val_loss}, with best epoch {best_epoch}")
       
                
    learning_rates.append(scheduler.get_last_lr()[0])

    log.info("Storing the training artifacts detials")
    
    data = list(zip(train_losses, val_losses))
    df = pd.DataFrame(data, columns=["train_losses", "val_losses"])
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
    

