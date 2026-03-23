import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from torch import optim
from torch.nn import BCEWithLogitsLoss

from src.load_data import load_data
from src.unet_model import UNet
from utils.config import Config
from src.trainning import train_and_val

def main():
    train_loader, val_loader = load_data()
    
    model = UNet(in_channels=Config.IN_CHANNELS, num_classes=Config.NUM_CLASSES)
    model = model.to(Config.DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=Config.LR)
    critirion = BCEWithLogitsLoss()

    model = train_and_val(model,
                          train_loader,
                          val_loader,
                          optimizer,
                          critirion,
                          30,
                          Config.DEVICE)
    

if __name__ == "__main__":
    main()