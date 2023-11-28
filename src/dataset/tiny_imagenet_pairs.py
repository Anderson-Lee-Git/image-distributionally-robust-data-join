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
    
    def __len__(self):
        return len(self.md)
    
    def __getitem__(self, index):
        row = self.md.iloc[index]
        image_1 = PIL.Image.open()
        image_2 = PIL.Image.open()
    
    def _get_md(self):
        return pd.read_csv(config("TINYIMAGENET_PAIRS_META_PATH"))

    def _get_path(self):
        return config("TINYIMAGENET_TRAIN_PATH")
    
        