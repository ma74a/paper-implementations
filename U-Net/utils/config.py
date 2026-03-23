from torchvision import transforms
import torch

class Config:
    # Data paths
    IMAGES = "/home/etman/etman/python/projects/paper-implementations/U-Net/data/images"
    MASKS  = "/home/etman/etman/python/projects/paper-implementations/U-Net/data/masks"

    # Hyperparameters
    LR         = 0.0001
    IMG_SIZE   = 256
    EPOCHS     = 50
    BATCH_SIZE = 16

    # Some values
    NUM_CLASSES = 1
    IN_CHANNELS = 3

    # Device
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    # Image transforms
    IMAGE_TRANSFORMS = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    # Mask transforms
    MASK_TRANSFORMS = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE),
                          interpolation=transforms.InterpolationMode.NEAREST),
        transforms.PILToTensor(),
    ])