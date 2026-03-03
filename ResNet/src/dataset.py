import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


import torch
from torch.utils.data import Dataset

from PIL import Image
import matplotlib.pyplot as plt

from utils.config import Config


class CustomDataset(Dataset):
    def __init__(self, data_dir, transforms=None):
        self.data_dir = data_dir
        self.transforms = transforms
        
        self.images_paths = []
        self.class_to_idx = {}
        self.labels = []
        self.classes = []

        self.__load_images()

    def __load_images(self):
        for label, class_name in enumerate(sorted(os.listdir(self.data_dir))):
            class_dir = os.path.join(self.data_dir, class_name)
            self.class_to_idx[class_name] = label
            self.classes.append(class_name)

            for img_name in os.listdir(class_dir):
                if img_name.endswith((".jpg", ".png", ".jpeg")):
                    img_path = os.path.join(class_dir, img_name)
                    self.images_paths.append(img_path)
                    self.labels.append(label)

    def __len__(self):
        return len(self.images_paths)
    
    def __getitem__(self, idx):
        img = self.images_paths[idx]
        label = self.labels[idx]

        img = Image.open(img).convert("RGB")

        if self.transforms:
            img = self.transforms(img)

        return img, label
    



# if __name__ == "__main__":
#     obj = CustomDataset(data_dir=Config.TRAIN_DATA, transforms=Config.TRANSFORMS["train"])
#     # img, label = obj[0]
#     for i in range(3):
#         img, label = obj[i]
#         print(img.shape)
#     # print(img)
#     plt.imshow(img.permute(1, 2, 0))
#     plt.show()