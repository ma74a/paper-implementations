from torch.utils.data import DataLoader

import matplotlib.pyplot as plt
from typing import List
import numpy as np


def show_batch(train_loader: DataLoader) -> None:
    batch = next(iter(train_loader))
    images, labels = batch

    for img, mask in zip(images[:8], labels[:8]):
        plt.figure(figsize=(11, 5))

        plt.subplot(1, 2, 1)
        plt.imshow(np.transpose(img.cpu().numpy(), (1, 2, 0)))
        plt.title("Image")
        plt.axis("off")

        plt.subplot(1, 2, 2)
        plt.imshow(mask.cpu().numpy())
        plt.title("Mask")
        plt.axis("off")

        plt.tight_layout()
        plt.show()