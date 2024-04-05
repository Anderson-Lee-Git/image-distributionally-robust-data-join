import os

import PIL
import torch
from torch.utils.data import Dataset
from torch import nn
import pandas as pd
from decouple import Config, RepositoryEnv
config = Config(RepositoryEnv(".env"))

class CIFAR100Pairs(Dataset):
    def __init__(self, transform=None, subset=1.0, unbalanced=False) -> None:
        super().__init__()
        self.transform = transform
        self.unbalanced = unbalanced
        self.path = self._get_path()
        self.md = self._get_md()
        self._normalize_dist_weight()
        self.subset = subset
    
    def __len__(self):
        return int(len(self.md) * self.subset)
    
    def __getitem__(self, index):
        row = self.md.iloc[index]
        path_1 = os.path.join(os.path.join(self.path, str(row["label_1"])), row["id_1"])
        path_2 = os.path.join(os.path.join(self.path, str(row["label_2"])), row["id_2"])
        image_1 = PIL.Image.open(path_1).convert("RGB")
        image_2 = PIL.Image.open(path_2).convert("RGB")
        label = row["label_1"]  # label 2 is only for study usage
        aux = row["aux_2"]
        dist_weight = row["dist_weight"]
        if self.transform:
            image_1 = self.transform(image_1)
            image_2 = self.transform(image_2)
        sample = {
            "image_1": image_1,
            "image_2": image_2,
            "aux": aux,
            "dist_weight": dist_weight,
            "label": label
        }
        return sample
    
    def get_len_per_group(self):
        md = pd.read_csv(os.path.join(config("DATASET_ROOT"), config("CIFAR100_TRAIN_META_PATH")))
        return len(md.loc[md["group"] == 1]), len(md.loc[md["group"] == 2])
    
    def _get_md(self):
        if self.unbalanced:
            return pd.read_csv(os.path.join(config("DATASET_ROOT"), config("CIFAR100_PAIRS_UNBALANCED_META_PATH")))
        else:
            return pd.read_csv(os.path.join(config("DATASET_ROOT"), config("CIFAR100_PAIRS_META_PATH")))
        
    def _normalize_dist_weight(self):
        dist = self.md["dist"].to_numpy()
        normalized_dist = dist / dist.sum()
        weight = 1 / normalized_dist
        normalized_weight = weight / weight.sum()
        self.md.insert(loc=len(self.md.columns),
                       column="dist_weight",
                       value=normalized_weight,
                       allow_duplicates=True)

    def _get_path(self):
        return os.path.join(config("DATASET_ROOT"), config("CIFAR100_TRAIN_PATH"))

    @staticmethod
    def collate_fn(batched_samples):
        assert len(batched_samples) > 0
        batch = {}
        batch["image_1"] = torch.stack([sample["image_1"] for sample in batched_samples], dim=0)
        batch["image_2"] = torch.stack([sample["image_2"] for sample in batched_samples], dim=0)
        batch["label"] = torch.stack([torch.tensor(sample["label"]) for sample in batched_samples], dim=0)
        batch["aux"] = nn.functional.one_hot(
            torch.stack([torch.tensor(sample["aux"]) for sample in batched_samples], dim=0),
            num_classes=20
        )
        batch["dist_weight"] = torch.stack([torch.tensor(sample["dist_weight"]) for sample in batched_samples], dim=0)
        return batch
    