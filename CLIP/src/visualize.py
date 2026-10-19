import math
import textwrap

import matplotlib.pyplot as plt
import numpy as np
import torch

CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)


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