import os

import PIL
from torch.utils.data import Dataset
import torch
import pandas as pd
from torchvision import transforms
import json
import numpy as np
from decouple import Config, RepositoryEnv
config = Config(RepositoryEnv(".env"))

class CIFAR100_C(Dataset):
    def __init__(self, transform, corruption: str, severity: int) -> None:
        super().__init__()
        self.path = self._get_path()
        self.images, self.labels = self._get_data(corruption, severity)
        self.aux_map = json.load(open(os.path.join(config("DATASET_ROOT"), config("CIFAR100_AUX_MAP")), "r"))
        self.transform = transform
        self._adapt_transforms()
    
    def __len__(self):
        return self.images.shape[0]
    
    def __getitem__(self, index):
        image = self.transform(self.images[index])
        label = torch.tensor(self.labels[index])
        aux = torch.tensor(int(self.aux_map[str(self.labels[index])]))
        sample = {
            "image": image,
            "label": label,
            "aux": aux
        }
        return sample

    def _adapt_transforms(self):
        """
        Add to PIL because cifar100_C comes in numpy array type
        """
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            self.transform
        ])
    
    def _get_data(self, corruption, severity=1):
        if corruption not in self._get_supported_corruptions() and \
            corruption != "all":
            raise NotImplementedError(f"corruption {corruption} not supported")
        
        if corruption != "all":
            path = os.path.join(self.path, corruption + ".npy")
            data = np.load(path)
            path = os.path.join(self.path, "labels.npy")
            labels = np.load(path)
            if severity == 1:
                return data[:10000], labels[:10000]
            else:
                return data[-10000:], labels[-10000:]
        else:
            # load all corruptions
            total_data = None
            total_labels = None
            for c in self._get_supported_corruptions():
                if total_data is None:
                    total_data, total_labels = self._get_data(c, severity=severity)
                else:
                    data, labels = self._get_data(c, severity=severity)
                    total_data = np.concatenate([total_data, data], axis=0)
                    total_labels = np.concatenate([total_labels, labels], axis=0)
            return total_data, total_labels
    
    def _get_path(self):
        return os.path.join(config("DATASET_ROOT"), config("CIFAR100_C_PATH"))
    
    def _get_supported_corruptions(self):
        return [
            "brightness",
            "contrast",
            "defocus_blur",
            "elastic_transform",
            "fog",
            "frost",
            "gaussian_blur",
            "gaussian_noise",
            "glass_blur",
            "impulse_noise",
            "jpeg_compression",
            "motion_blur",
            "pixelate",
            "saturate",
            "shot_noise",
            "snow",
            "spatter",
            "speckle_noise",
            "zoom_blur"
        ]