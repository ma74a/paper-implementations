import torch
from torch.utils.data import DataLoader

import matplotlib.pyplot as plt
from typing import List
import numpy as np
from pathlib import Path


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
        
        
def plot_confusion_matrix(
    confusion_matrix: torch.Tensor,
    class_names: list[str],
    normalize: bool = True,
    save_path: str | None = None,
    show: bool = False,
):
    """
    Plot semantic segmentation confusion matrix.

    Rows: Ground-truth classes.
    Columns: Predicted classes.

    Args:
        confusion_matrix: [C, C] confusion matrix.
        class_names: Names of segmentation classes.
        normalize: Normalize each row.
        save_path: Optional path to save the figure.
        show: Whether to display the figure.
    """

    cm = confusion_matrix.detach().cpu().numpy().astype(
        np.float64
    )

    num_classes = cm.shape[0]

    if cm.shape != (num_classes, num_classes):
        raise ValueError("Confusion matrix must be square.")

    if len(class_names) != num_classes:
        raise ValueError(
            "Number of class names must match number of classes."
        )

    # Normalize each ground-truth class row
    if normalize:
        row_sums = cm.sum(axis=1, keepdims=True)

        cm = np.divide(
            cm,
            row_sums,
            out=np.zeros_like(cm),
            where=row_sums != 0,
        )

    fig, ax = plt.subplots(figsize=(14, 12))

    im = ax.imshow(
        cm,
        interpolation="nearest",
        cmap="Blues",
    )

    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    ax.set(
        xticks=np.arange(num_classes),
        yticks=np.arange(num_classes),
        xticklabels=class_names,
        yticklabels=class_names,
        xlabel="Predicted Class",
        ylabel="Ground Truth Class",
        title="SegFormer Confusion Matrix",
    )

    plt.setp(
        ax.get_xticklabels(),
        rotation=90,
        ha="center",
    )

    # Annotate small confusion matrices
    if num_classes <= 10:
        for i in range(num_classes):
            for j in range(num_classes):

                value = cm[i, j]

                text_value = (
                    f"{value:.2f}"
                    if normalize
                    else f"{value:.0f}"
                )

                ax.text(
                    j,
                    i,
                    text_value,
                    ha="center",
                    va="center",
                    color="black",
                    fontsize=8,
                )

    fig.tight_layout()

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        fig.savefig(
            save_path,
            dpi=300,
            bbox_inches="tight",
        )

    if show:
        plt.show()

    plt.close(fig)
