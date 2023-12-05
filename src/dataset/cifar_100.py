import os

import PIL
from torch.utils.data import Dataset
import pandas as pd
from torchvision import transforms
from decouple import Config, RepositoryEnv
config = Config(RepositoryEnv(".env"))

class CIFAR100(Dataset):
    def __init__(self, train_transform=None, val_transform=None, split='train', subset=1.0, group=1,
                include_path=False, include_origin=False) -> None:
        super().__init__()
        self.split = split
        self.train_transform = train_transform
        self.val_transform = val_transform
        self.group = group
        self.path = self._get_path()
        self.md = self._get_md()
        self.subset = subset
        self.include_path = include_path
        self.include_origin = include_origin
    
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
            return self._get_custom_item(self.minimum_transform(), image, label, img_path)

    
    def minimum_transform(self):
        transform_train = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5071, 0.4867, 0.4408], std=[0.2675, 0.2565, 0.2761])])
        return transform_train
    
    def _get_custom_item(self, transform, image, label, img_path):
        basic_transform = self.minimum_transform()
        if transform:
            transformed_image = transform(image)
        else:
            transformed_image = basic_transform(image)
        # basic processing for original image
        image = basic_transform(image)
        if self.include_origin and self.include_path:
            return transformed_image, label, image, img_path
        elif self.include_origin:
            return transformed_image, label, image
        elif self.include_path:
            return transformed_image, label, img_path
        else:
            return transformed_image, label
    
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
            md = df.loc[df["group"] == self.group]
        elif self.split == 'val':
            md = pd.read_csv(config("CIFAR100_VAL_META_PATH"))
        else:
            md = pd.read_csv(config("CIFAR100_TEST_META_PATH"))
        assert len(md) > 0
        return md