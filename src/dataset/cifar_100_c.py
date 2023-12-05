import os

import PIL
from torch.utils.data import Dataset
import pandas as pd
from torchvision import transforms
import numpy as np
from decouple import Config, RepositoryEnv
config = Config(RepositoryEnv(".env"))

class CIFAR100_C(Dataset):
    def __init__(self, corruption: str, severity: int) -> None:
        super().__init__()
        self.path = self._get_path()
        self.images, self.labels = self._get_data(corruption, severity)
        self.transform = self.minimum_transform()
    
    def __len__(self):
        return self.images.shape[0]
    
    def __getitem__(self, index):
        image = self.images[index]
        label = self.labels[index]
        return self.transform(image), label
    
    def minimum_transform(self):
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5071, 0.4867, 0.4408], std=[0.2675, 0.2565, 0.2761])])
        return transform
    
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
        return config("CIFAR100_C_PATH")
    
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