import os

import PIL
import torch
from torch.utils.data import Dataset
import pandas as pd
from decouple import Config, RepositoryEnv
config = Config(RepositoryEnv(".env"))

class TinyImagenetPairs(Dataset):
    def __init__(self, transform=None, subset=1.0) -> None:
        super().__init__()
        self.transform = transform
        self.path = self._get_path()
        self.md = self._get_md()
        self.subset = subset
    
    def __len__(self):
        return int(len(self.md) * self.subset)
    
    def __getitem__(self, index):
        row = self.md.iloc[index]
        path_1 = os.path.join(os.path.join(self.path, row["class_1"]), row["id_1"])
        path_2 = os.path.join(os.path.join(self.path, row["class_2"]), row["id_2"])
        image_1 = PIL.Image.open(path_1).convert("RGB")
        image_2 = PIL.Image.open(path_2).convert("RGB")
        label = row["label_1"]  # label 2 is only for study usage
        if self.transform:
            image_1 = self.transform(image_1)
            image_2 = self.transform(image_2)
        return image_1, image_2, label
    
    def get_len_per_group(self):
        md = pd.read_csv(config("TINYIMAGENET_TRAIN_META_PATH"))
        return len(md.loc[md["group"] == 1]), len(md.loc[md["group"] == 2])
    
    def _get_md(self):
        return pd.read_csv(config("TINYIMAGENET_PAIRS_META_PATH"))

    def _get_path(self):
        return config("TINYIMAGENET_TRAIN_PATH")
    
        