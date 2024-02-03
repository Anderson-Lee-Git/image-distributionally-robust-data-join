import os

import PIL
from torch.utils.data import Dataset
import pandas as pd
from torchvision import transforms
from decouple import Config, RepositoryEnv
config = Config(RepositoryEnv(".env"))

class CIFAR100(Dataset):
    def __init__(self, train_transform=None, 
                 minimum_transform=None,
                 val_transform=None, split='train', subset=1.0, group=0,
                 include_path=False) -> None:
        super().__init__()
        self.split = split
        self.train_transform = train_transform
        self.val_transform = val_transform
        self.minimum_transform = minimum_transform
        self.group = group
        self.path = self._get_path()
        self.md = self._get_md()
        self.subset = subset
        self.include_path = include_path
    
    def __len__(self):
        return int(len(self.md) * self.subset)
    
    def __getitem__(self, index):
        row = self.md.iloc[index]
        label = None
        img_path = os.path.join(os.path.join(self.path, str(row["label"])), row["id"])
        label = row["label"]
        image = PIL.Image.open(img_path)
        image = image.convert("RGB")
        if self.split == 'train':
            return self._get_custom_item(self.train_transform, image, label, img_path)
        elif self.split == "val":
            return self._get_custom_item(self.val_transform, image, label, img_path)
        else:
            return self._get_custom_item(self.minimum_transform, image, label, img_path)
    
    def _get_custom_item(self, transform, image, label, img_path):
        if transform:
            transformed_image = transform(image)
        else:
            transformed_image = self.minimum_transform(image)
        # basic processing for original image
        image = self.minimum_transform(image)
        if self.include_path:
            return transformed_image, image, label, img_path
        else:
            return transformed_image, image, label

    def _get_path(self):
        if self.split == 'train':
            return config("CIFAR100_TRAIN_PATH")
        elif self.split == 'val':
            return config("CIFAR100_VAL_PATH")
        else:
            return config("CIFAR100_TEST_PATH")
    
    def _get_md(self):
        if self.split == 'train':
            df = pd.read_csv(config("CIFAR100_TRAIN_META_PATH"))
            if self.group != 0:
                md = df.loc[df["group"] == self.group]
            else:
                md = df
        elif self.split == 'val':
            md = pd.read_csv(config("CIFAR100_VAL_META_PATH"))
        else:
            md = pd.read_csv(config("CIFAR100_TEST_META_PATH"))
        assert len(md) > 0
        return md