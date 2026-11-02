
import torch


class MeanIoU:
    """
    Mean Intersection over Union for semantic segmentation.

    Accumulates a confusion matrix across batches.

    Args:
        num_classes: Number of segmentation classes.
        ignore_index: Label to ignore during evaluation.
        device: Device used for metric calculations.
    """

    def __init__(
        self,
        num_classes: int,
        ignore_index: int = 255,
        device: str = "cpu",
    ):
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.device = torch.device(device)

        self.confusion_matrix = torch.zeros(
            (num_classes, num_classes),
            dtype=torch.int64,
            device=self.device,
        )

    @torch.no_grad()
    def update(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
    ) -> None:
        """
        Update confusion matrix using a batch.

        Args:
            predictions:
                Logits [B, C, H, W] or class IDs [B, H, W].

            targets:
                Ground-truth class IDs [B, H, W].
        """

        # Convert logits to predicted class IDs
        if predictions.ndim == targets.ndim + 1:
            predictions = predictions.argmax(dim=1)

        if predictions.shape != targets.shape:
            raise ValueError(
                f"Shape mismatch: predictions {predictions.shape}, "
                f"targets {targets.shape}"
            )

        predictions = predictions.to(
            device=self.device,
            dtype=torch.int64,
        ).flatten()

        targets = targets.to(
            device=self.device,
            dtype=torch.int64,
        ).flatten()

        # Remove ignored and invalid pixels
        valid_mask = (
            (targets != self.ignore_index)
            & (targets >= 0)
            & (targets < self.num_classes)
            & (predictions >= 0)
            & (predictions < self.num_classes)
        )

        predictions = predictions[valid_mask]
        targets = targets[valid_mask]

        # Encode target-prediction pairs
        indices = (
            targets * self.num_classes + predictions
        )

        # Calculate batch confusion matrix
        batch_confusion = torch.bincount(
            indices,
            minlength=self.num_classes ** 2,
        ).reshape(
            self.num_classes,
            self.num_classes,
        )

        self.confusion_matrix += batch_confusion

    @torch.no_grad()
    def compute(self) -> dict:
        """
        Calculate per-class IoU and mean IoU.

        Classes with zero union are excluded from the mean.
        """

        tp = torch.diag(
            self.confusion_matrix
        ).float()

        fp = (
            self.confusion_matrix.sum(dim=0).float() - tp
        )

        fn = (
            self.confusion_matrix.sum(dim=1).float() - tp
        )

        union = tp + fp + fn

        # Exclude classes absent from both prediction and target
        valid_classes = union > 0

        per_class_iou = torch.full_like(
            union,
            float("nan"),
        )

        per_class_iou[valid_classes] = (
            tp[valid_classes] / union[valid_classes]
        )

        miou = per_class_iou[valid_classes].mean()

        return {
            "miou": miou.item(),
            "per_class_iou": per_class_iou.cpu(),
            "confusion_matrix": self.confusion_matrix.cpu(),
        }

    def reset(self) -> None:
        """Reset accumulated statistics."""

        self.confusion_matrix.zero_()
