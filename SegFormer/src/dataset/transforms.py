import random
import numpy as np
import torch
import torchvision.transforms as T
import torchvision.transforms.functional as TF

MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]
IGNORE = 255


class TrainTransform:
    def __init__(self, crop=512):
        self.crop = crop

    def __call__(self, img, mask):
        # 1. random resize (0.5x to 2x)
        scale = random.uniform(0.5, 2.0)
        w, h = img.size
        new_w, new_h = int(w * scale), int(h * scale)
        img = TF.resize(img, (new_h, new_w))
        mask = TF.resize(mask, (new_h, new_w), interpolation=T.InterpolationMode.NEAREST)

        # 2. pad if smaller than the crop size
        pad_w = max(self.crop - new_w, 0)
        pad_h = max(self.crop - new_h, 0)
        img = TF.pad(img, [0, 0, pad_w, pad_h], fill=0)
        mask = TF.pad(mask, [0, 0, pad_w, pad_h], fill=IGNORE)

        # 3. random crop (same box for image and mask)
        i, j, h, w = T.RandomCrop.get_params(img, (self.crop, self.crop))
        img = TF.crop(img, i, j, h, w)
        mask = TF.crop(mask, i, j, h, w)

        # 4. random horizontal flip
        if random.random() < 0.5:
            img = TF.hflip(img)
            mask = TF.hflip(mask)

        # to tensors
        img = TF.normalize(TF.to_tensor(img), MEAN, STD)
        mask = torch.from_numpy(np.array(mask)).long()
        return img, mask


class ValTransform:
    def __init__(self, size=512):
        self.size = size

    def __call__(self, img, mask):
        img = TF.resize(img, (self.size, self.size))
        mask = TF.resize(mask, (self.size, self.size), interpolation=T.InterpolationMode.NEAREST)
        img = TF.normalize(TF.to_tensor(img), MEAN, STD)
        mask = torch.from_numpy(np.array(mask)).long()
        return img, mask