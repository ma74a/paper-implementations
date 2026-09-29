from torch.utils.data import Dataset
from PIL import Image
import os

class TomAndJerryDataset(Dataset):
    def __init__(self, data_dir, transforms=None):
        self.data_dir = data_dir
        self.transforms = transforms

        self.class_to_idx = {}
        self.classes = []
        self.images_path = []
        self.labels = []

        self._load_images()

    def _load_images(self):
        for label, cls_name in enumerate(sorted(os.listdir(self.data_dir))):
            cls_dir = os.path.join(self.data_dir, cls_name)
            self.classes.append(cls_name)
            self.class_to_idx[cls_name] = label
            for img in os.listdir(cls_dir):
                if img.lower().endswith(('.jpg', '.png', '.jpeg')):
                    img_path = os.path.join(cls_dir, img)
                    self.images_path.append(img_path)
                    self.labels.append(label)

    def __len__(self):
        return len(self.images_path)

    def __getitem__(self, idx):
        img_path = self.images_path[idx]
        label = self.labels[idx]

        img = Image.open(img_path).convert("RGB")

        if self.transforms:
            img = self.transforms(img)

        return img, label

    

