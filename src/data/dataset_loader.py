import os
import cv2
import torch
from torch.utils.data import Dataset


class ImageMaskDataset(Dataset):
    def __init__(self, root, transforms=None):
        self.img_dir = os.path.join(root, "images")
        self.mask_dir = os.path.join(root, "masks")
        self.files = sorted(os.listdir(self.img_dir))
        self.transforms = transforms

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        name = self.files[idx]

        img = cv2.imread(os.path.join(self.img_dir, name), 0)
        mask = cv2.imread(os.path.join(self.mask_dir, name), 0)

        img = img.astype("float32") / 255.0
        mask = (mask > 128).astype("float32")

        img = torch.tensor(img).unsqueeze(0)
        mask = torch.tensor(mask).unsqueeze(0)

        if self.transforms:
            img, mask = self.transforms(img, mask)

        return img, mask
