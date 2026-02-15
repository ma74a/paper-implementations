import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn

from src.model import AlexNet
from src.load_data import load_data
from src.training import train_and_val
from utils.config import Config
from utils.visualize import plot_curves

def main():
    train_loader, val_loader = load_data()

    model = AlexNet(num_classes=Config.NUM_CLASSES)
    model = model.to(Config.DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=Config.LEARNING_RATE, momentum=0.9, weight_decay=0.0005)

    model, train_losses, train_accuracies, val_losses, val_accuracies = train_and_val(model,
                                                                                      train_loader,
                                                                                      val_loader,
                                                                                      criterion,
                                                                                      optimizer,
                                                                                      Config.EPOCHS)
    

    plot_curves(train_losses, val_losses)


if __name__ == "__main__":
    main()