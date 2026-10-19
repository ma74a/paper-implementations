import torch
from torch.nn import functional as F


def clip_contrastive_loss(
    logits_per_image: torch.Tensor,
    logits_per_text: torch.Tensor
) -> torch.Tensor:
    """Calculate CLIP's symmetric image-text contrastive loss."""
    # the i-th image matches the i-th text, so the correct class is the diagonal
    batch_size = logits_per_image.shape[0]
    
    labels = torch.arange(batch_size, device=logits_per_image.device)
    print(labels)
    
    image_loss = F.cross_entropy(logits_per_image, labels)
    text_loss = F.cross_entropy(logits_per_text, labels)
    
    # symmetric oss
    loss = (image_loss + text_loss) / 2
    
    return loss

