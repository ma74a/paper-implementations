import math
import textwrap

import matplotlib.pyplot as plt
import numpy as np
import torch
import os


CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)

def saving_training_plots(history_df, lr, output_dir):
    """
    Save train vs. val loss and the learning rate curve.
 
    Input : history_df with columns 'train loss' and 'val loss' (one row per epoch)
            lr         list of learning rates, one value per training step
            output_dir folder where training_curves.png is saved
    Output : None
    """
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(1, 2, figsize=(12, 5))
 
    ax[0].plot(history_df["train_losses"], label="Train Loss")
    ax[0].plot(history_df["val_losses"], label="Validation Loss")
    ax[0].set_title("Loss Curves")
    ax[0].set_xlabel("Epoch")
    ax[0].set_ylabel("Loss")
    ax[0].legend()
 
    ax[1].plot(lr, label="Learning rate per step")
    ax[1].set_title("Learning Rate Curve")
    ax[1].set_xlabel("Step")
    ax[1].set_ylabel("Learning rate")
    ax[1].legend()
 
    fig.tight_layout()
    plot_path = os.path.join(output_dir, "training_curves.png")
    fig.savefig(plot_path, dpi=150)
    plt.close(fig)  # free the figure so it doesn't pile up in memory or show twice
    # log.info(f"Saved training plot to {plot_path}")


def denormalize(
    images: torch.Tensor,
    mean: tuple = CLIP_MEAN,
    std: tuple = CLIP_STD,
) -> torch.Tensor:
    # images [B, 3, H, W], undo (x - mean) / std so the colors look normal again
    mean = torch.tensor(mean, device=images.device).view(1, 3, 1, 1)
    std = torch.tensor(std, device=images.device).view(1, 3, 1, 1)
    return (images * std + mean).clamp(0, 1)


def show_batch(
    images: torch.Tensor,
    texts: torch.Tensor,
    tokenizer,
    num_images: int = 8,
    cols: int = 4,
    wrap_width: int = 30,
    save_path: str = None,
) -> None:
    """
    images: [B, 3, H, W] normalized images from the DataLoader
    texts:  [B, 77] token ids from the DataLoader
    """
    num_images = min(num_images, images.shape[0])
    rows = math.ceil(num_images / cols)

    images = denormalize(images[:num_images].detach().cpu())

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3.5, rows * 3.8))
    axes = np.atleast_1d(axes).flatten()

    for i, ax in enumerate(axes):
        ax.axis("off")
        if i >= num_images:
            continue  # empty slot in the last row

        # [3, H, W] -> [H, W, 3] for matplotlib
        ax.imshow(images[i].permute(1, 2, 0).numpy())

        # skip_special_tokens removes <|startoftext|>, <|endoftext|> and the padding
        caption = tokenizer.decode(texts[i].tolist(), skip_special_tokens=True)
        ax.set_title("\n".join(textwrap.wrap(caption, wrap_width)), fontsize=9)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
    else:
        plt.show()


if __name__ == "__main__":
    # usage with the objects from train.py
    # images, texts = next(iter(train_loader))
    # show_batch(images, texts, tokenizer, num_images=8, cols=4)
    pass