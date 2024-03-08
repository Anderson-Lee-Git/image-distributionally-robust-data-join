import os

import PIL
import torch
from torch.utils.data import Dataset
import pandas as pd
from torchvision import transforms
from decouple import Config, RepositoryEnv
config = Config(RepositoryEnv(".env"))

class CelebAPairs(Dataset):
    def __init__(self, transform=None, subset=1.0, unbalanced=False) -> None:
        super().__init__()
        self.transform = transform
        self.unbalanced = unbalanced
        self.path = self._get_path()
        self.md = self._get_md()
        self.subset = subset
        self._adapt_transforms()
    
    def __len__(self):
        return int(len(self.md) * self.subset)
    
    def __getitem__(self, index):
        row = self.md.iloc[index]
        path_1 = os.path.join(self.path, row["id_1"])
        path_2 = os.path.join(self.path, row["id_2"])
        image_1 = PIL.Image.open(path_1).convert("RGB")
        image_2 = PIL.Image.open(path_2).convert("RGB")
        label = row["label_1"]  # label 2 is only for study usage
        aux = row["aux_2"]
        if aux == -1:
            aux = torch.tensor(0)
        if label == -1:
            label = torch.tensor(0)
        if self.transform:
            image_1 = self.transform(image_1)
            image_2 = self.transform(image_2)
        sample = {
            "image_1": image_1,
            "image_2": image_2,
            "aux": aux,
            "label": label
        }
        return sample
    
    def get_len_per_group(self):
        if self.unbalanced:
            md = pd.read_csv(os.path.join(config("DATASET_ROOT"), config("CELEB_A_TRAIN_UNBALANCED_META_PATH")))
        else:
            md = pd.read_csv(os.path.join(config("DATASET_ROOT"), config("CELEB_A_TRAIN_META_PATH")))
        return len(md.loc[md["group"] == 1]), len(md.loc[md["group"] == 2])
    
    def _get_md(self):
        if self.unbalanced:
            return pd.read_csv(os.path.join(config("DATASET_ROOT"), config("CELEB_A_PAIRS_UNBALANCED_META_PATH")))
        else:
            return pd.read_csv(os.path.join(config("DATASET_ROOT"), config("CELEB_A_PAIRS_META_PATH")))

    def _get_path(self):
        return os.path.join(config("DATASET_ROOT"), config("CELEB_A_PATH"))
    
    def _adapt_transforms(self):
        """
        Add center crop by the width=178 of celebA because
        celebA comes in shape: (178, 218)
        """
        self.transform = transforms.Compose([
            transforms.CenterCrop(178),
            self.transform
        ])

    @staticmethod
    def collate_fn(batched_samples):
        assert len(batched_samples) > 0
        batch = {}
        batch["image_1"] = torch.stack([sample["image_1"] for sample in batched_samples], dim=0)
        batch["image_2"] = torch.stack([sample["image_2"] for sample in batched_samples], dim=0)
        batch["label"] = torch.stack([torch.tensor(sample["label"]) for sample in batched_samples], dim=0)
        batch["aux"] = torch.stack([torch.tensor(sample["aux"]) for sample in batched_samples], dim=0)
        return batch
    