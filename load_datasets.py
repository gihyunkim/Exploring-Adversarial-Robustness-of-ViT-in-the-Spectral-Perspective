import torch
from PIL import Image
import numpy as np
from torch.utils.data import Dataset


class AdversarialTrainDataset(Dataset):
    def __init__(self, img_paths, labels, transform):
        self.transform = transform
        self.img_paths = img_paths
        self.labels = labels

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img = Image.open(self.img_paths[idx])
        img = self.transform(img)
        img = torch.clamp(img, 0, 1)
        label = np.array((self.labels[idx]-1))
        return img, torch.tensor(label)
