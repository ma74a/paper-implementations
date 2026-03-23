from torch.utils.data import Dataset
from torchvision import transforms

import os
from PIL import Image
import matplotlib.pyplot as plt

class SegmentationDataset(Dataset):
    def __init__(self, images_path,
                       masks_path,
                       img_transforms=None,
                       mask_transforms=None):
        self.images_path = images_path
        self.masks_path = masks_path
        self.img_transforms = img_transforms
        self.mask_transforms = mask_transforms

        # self.images = os.listdir(self.images_path)
        self.images = []
        self.masks = []
        for img in sorted(os.listdir(self.images_path)):
            img_path = os.path.join(self.images_path, img)
            self.images.append(img_path)
        # self.masks = os.listdir(self.masks_path)
        for mask in sorted(os.listdir(self.masks_path)):
            msk_path = os.path.join(self.masks_path, mask)
            self.masks.append(msk_path)

    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img, mask = self.images[idx], self.masks[idx]
        # print(img.shape)
        img = Image.open(img).convert("RGB")
        mask = Image.open(mask).convert("L")

        if self.img_transforms:
            img = self.img_transforms(img)
        if self.mask_transforms:
            mask = self.mask_transforms(mask)

        return img, mask
    
if __name__ == "__main__":
    IMAGE_TRANSFORMS = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor(),
        # transforms.Normalize(mean=[0.485, 0.456, 0.406],
        #                      std=[0.229, 0.224, 0.225]),
    ])

    # Mask transforms
    # - NEAREST interpolation: prevents blending between class indices
    # - PILToTensor: keeps values as integers (ToTensor would divide by 255)
    MASK_TRANSFORMS = transforms.Compose([
        transforms.Resize((512, 512),
                          interpolation=transforms.InterpolationMode.NEAREST),
        transforms.PILToTensor(),
    ])
    imgs = "/home/etman/etman/python/projects/paper-implementations/U-Net/data/images"
    masks = "/home/etman/etman/python/projects/paper-implementations/U-Net/data/masks"
    # images = os.listdir(imgs)
    # print(images)
    obj = SegmentationDataset(imgs, masks, IMAGE_TRANSFORMS, MASK_TRANSFORMS)
    # print(len(obj))
    i, m = obj[0]
    print(i.shape)
    print(m.shape)
    # plt.imshow(m.permute(1, 2, 0))
    # plt.show()
