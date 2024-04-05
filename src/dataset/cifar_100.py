import os

import PIL
import torch
from torch import nn
from torch.utils.data import Dataset
import pandas as pd
from decouple import Config, RepositoryEnv
config = Config(RepositoryEnv(".env"))

class CIFAR100(Dataset):
    def __init__(self, train_transform=None, 
                 minimum_transform=None,
                 val_transform=None, split='train', subset=1.0, group=0,
                 unbalanced=False,
                 include_path=False) -> None:
        super().__init__()
        self.split = split
        self.train_transform = train_transform
        self.val_transform = val_transform
        self.minimum_transform = minimum_transform
        self.group = group
        self.unbalanced = unbalanced
        self.path = self._get_path()
        self.md = self._get_md()
        self.subset = subset
        self.include_path = include_path
    
    def __len__(self):
        return int(len(self.md) * self.subset)
    
    def __getitem__(self, index):
        row = self.md.iloc[index]
        img_path = os.path.join(os.path.join(self.path, str(row["label"])), row["id"])
        label = torch.tensor(row["label"])
        aux = nn.functional.one_hot(torch.tensor(row["superclass"]), num_classes=20)
        image = PIL.Image.open(img_path)
        image = image.convert("RGB")
        if self.split == 'train':
            transform = self.train_transform
        elif self.split == "val":
            transform = self.val_transform
        else:
            transform = self.minimum_transform
        return self._get_custom_item(transform, image, label, aux, img_path)
    
    def _get_custom_item(self, transform, image, label, aux, img_path):
        if transform:
            transformed_image = transform(image)
        else:
            transformed_image = self.minimum_transform(image)
        # basic processing for original image
        image = self.minimum_transform(image)
        sample = {
            "image": transformed_image,
            "original_image": image,
            "label": label,
            "aux": aux,
            "path": img_path
        }
        return sample
    
    def _get_path(self):
        if self.split == 'train':
            return os.path.join(config("DATASET_ROOT"), config("CIFAR100_TRAIN_PATH"))
        elif self.split == 'val':
            return os.path.join(config("DATASET_ROOT"), config("CIFAR100_VAL_PATH"))
        else:
            return os.path.join(config("DATASET_ROOT"), config("CIFAR100_TEST_PATH"))
    
    def _get_md(self):
        if self.split == 'train':
            if self.unbalanced:
                df = pd.read_csv(os.path.join(config("DATASET_ROOT"), config("CIFAR100_TRAIN_UNBALANCED_META_PATH")))
            else:
                df = pd.read_csv(os.path.join(config("DATASET_ROOT"), config("CIFAR100_TRAIN_META_PATH")))
            if self.group != 0:
                md = df.loc[df["group"] == self.group]
            else:
                md = df
        elif self.split == 'val':
            md = pd.read_csv(os.path.join(config("DATASET_ROOT"), config("CIFAR100_VAL_META_PATH")))
        else:
            md = pd.read_csv(os.path.join(config("DATASET_ROOT"), config("CIFAR100_TEST_META_PATH")))
        assert len(md) > 0
        return md
    
    def set_test_sub_population(self, ratio):
        """
        :param ratio: a list of 20 numbers between 0 and 1 to determine the number of samples
        for each superclass
        """
        assert self.split == "test"
        md = pd.DataFrame(columns=self.md.columns)
        for i in range(len(ratio)):
            sub_df = self.md.loc[self.md["superclass"] == i].copy()
            md = pd.concat([md, sub_df.iloc[:int(ratio[i] * len(sub_df))].copy()])
        md = md.sample(frac=1).reset_index(drop=True)
        self.total_md = self.md.copy()
        self.md = md.copy()
        assert len(self.md) > 0
    
    def reset_test_population(self):
        assert hasattr(self, "total_md")
        self.md = self.total_md.copy()

class CIFAR100_A(CIFAR100):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
    
    def __getitem__(self, index):
        """
        Label of an item is superclass instead of fineclass
        """
        row = self.md.iloc[index]
        img_path = os.path.join(os.path.join(self.path, str(row["label"])), row["id"])
        label = torch.tensor(row["superclass"])
        aux = nn.functional.one_hot(torch.tensor(row["superclass"]), num_classes=20)
        image = PIL.Image.open(img_path)
        image = image.convert("RGB")
        if self.split == 'train':
            transform = self.train_transform
        elif self.split == "val":
            transform = self.val_transform
        else:
            transform = self.minimum_transform
        return self._get_custom_item(transform, image, label, aux, img_path)

