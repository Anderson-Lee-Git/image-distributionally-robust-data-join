import os
import pickle

import PIL
from torch.utils.data import Dataset
import pandas as pd
from torchvision import transforms
from decouple import Config, RepositoryEnv
config = Config(RepositoryEnv(".env"))

from utils.visualization import visualize_image

class AuxCIFAR100(Dataset):
    def __init__(self, train_transform=None, val_transform=None, split='train', subset=1.0) -> None:
        super().__init__()
        self.split = split
        self.train_transform = train_transform
        self.val_transform = val_transform
        self.path = self._get_path()
        self.md = self._get_md()
        self.subset = subset
        self.aux_set = self._get_aux_set()
    
    def __len__(self):
        if self.split == "train":
            return int((len(self.md) + len(self.aux_set["data"])) * self.subset)
        else:
            return int(len(self.md) * self.subset)
    
    def __getitem__(self, index):
        if self.split == "train":
            if index < len(self.md):
                row = self.md.iloc[index]
                img_path = os.path.join(os.path.join(self.path, str(row["label"])), row["id"])
                image = PIL.Image.open(img_path)
                image = image.convert("RGB")
            else:
                # image would be numpy.ndarray
                image = self.aux_set["data"][index-len(self.md)]
            if self.split == 'train':
                transform = self.train_transform
            elif self.split == "val":
                transform = self.val_transform
            else:
                transform = self.minimum_transform()
            return transform(image), self.minimum_transform()(image)
        else:
            row = self.md.iloc[index]
            img_path = os.path.join(os.path.join(self.path, str(row["label"])), row["id"])
            image = PIL.Image.open(img_path)
            image = image.convert("RGB")
            return self.val_transform(image), self.minimum_transform()(image)
    
    def minimum_transform(self):
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5071, 0.4867, 0.4408], std=[0.2675, 0.2565, 0.2761])])
        return transform
    
    def _get_path(self):
        if self.split == 'train':
            return config("CIFAR100_TRAIN_PATH")
        elif self.split == 'val':
            return config("CIFAR100_VAL_PATH")
        else:
            return config("CIFAR100_TEST_PATH")
    
    def _get_md(self):
        if self.split == 'train':
            md = pd.read_csv(config("CIFAR100_TRAIN_META_PATH"))
        elif self.split == 'val':
            md = pd.read_csv(config("CIFAR100_VAL_META_PATH"))
        else:
            md = pd.read_csv(config("CIFAR100_TEST_META_PATH"))
        assert len(md) > 0
        return md
    
    def _get_aux_set(self):
        file_handle = open(config("CIFAR100_PSEUDO_PATH"), "rb")
        data = pickle.load(file_handle)
        return data
            