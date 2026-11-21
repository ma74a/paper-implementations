
from torchvision import datasets, transforms
from torch.utils.data import DataLoader



transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0,), (1,))  # normalise to range [0, 1]
])


def load_data():
    train_dataset = datasets.MNIST(
        root="./data",
        train=True,
        transform=transform,
        download=True
    )
    test_dataset = datasets.MNIST(
        root="./data",
        train=False,
        transform=transform,
        download=True
    )
    
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=16,
        shuffle=True
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=16,
        shuffle=False
    )
    
    return train_dataset, test_dataset, train_loader, test_loader


# if __name__ == "__main__":
#     load_data()