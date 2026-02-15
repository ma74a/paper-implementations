import torch
from torchvision import transforms

class Config:
    TRAIN_DATA = "/your/path"
    VAL_DATA = "/your/path"

    # Hyperparameters
    LEARNING_RATE = 0.01
    BATCH_SIZE = 32
    EPOCHS = 100

    IMG_SIZE = 256
    NUM_CLASSES = 5

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    TRANSFORMS = {
        "train": transforms.Compose([
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.RandomCrop((227, 227)),
            transforms.RandomHorizontalFlip(0.5),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                         std=[0.229, 0.224, 0.225])
        ]),
        "val": transforms.Compose([
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.CenterCrop((227, 227)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                         std=[0.229, 0.224, 0.225])
        ])
    }

