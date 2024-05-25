import os

import PIL
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import pandas as pd
from decouple import Config, RepositoryEnv
config = Config(RepositoryEnv(".env"))

class CelebA(Dataset):
    def __init__(self, train_transform=None,
                 minimum_transform=None,
                 val_transform=None,
                 split='train',
                 subset=1.0,
                 group=0,
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
        self._adapt_transforms()
    
    def __len__(self):
        return int(len(self.md) * self.subset)
    
    def __getitem__(self, index):
        row = self.md.iloc[index]
        img_path = os.path.join(self.path, str(row["image_id"]))
        label = torch.tensor(row["Blond_Hair"])
        aux = torch.tensor(row["Male"])
        if aux == -1:
            aux = torch.tensor(0)
        if label == -1:
            label = torch.tensor(0)
        aux = torch.nn.functional.one_hot(aux, num_classes=2)
        image = PIL.Image.open(img_path)
        image = image.convert("RGB")
        if self.split == 'train':
            transform = self.train_transform
        elif self.split == "val":
            transform = self.val_transform
        else:
            transform = self.minimum_transform
        return self._get_custom_item(transform, image, label, aux, img_path)
    
    def _adapt_transforms(self):
        """
        Add center crop by the width=178 of celebA because
        celebA comes in shape: (178, 218)
        """
        self.train_transform = transforms.Compose([
            transforms.CenterCrop(178),
            self.train_transform
        ])
        self.val_transform = transforms.Compose([
            transforms.CenterCrop(178),
            self.val_transform
        ])
        self.minimum_transform = transforms.Compose([
            transforms.CenterCrop(178),
            self.minimum_transform
        ])
    
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
        return os.path.join(config("DATASET_ROOT"), config("CELEB_A_PATH"))
    
    def _get_md(self):
        if self.split == 'train':
            if self.unbalanced:
                df = pd.read_csv(os.path.join(config("DATASET_ROOT"), config("CELEB_A_TRAIN_UNBALANCED_META_PATH")))
            else:
                df = pd.read_csv(os.path.join(config("DATASET_ROOT"), config("CELEB_A_TRAIN_META_PATH")))
            if self.group != 0:
                md = df.loc[df["group"] == self.group]
            else:
                md = df
        elif self.split == 'val':
            md = pd.read_csv(os.path.join(config("DATASET_ROOT"), config("CELEB_A_VAL_META_PATH")))
        else:
            md = pd.read_csv(os.path.join(config("DATASET_ROOT"), config("CELEB_A_TEST_META_PATH")))
        assert len(md) > 0
        return md