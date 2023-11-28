import os

import PIL
from torch.utils.data import Dataset
import pandas as pd
from torchvision import transforms
from decouple import Config, RepositoryEnv
config = Config(RepositoryEnv(".env"))

def simple_transform(args):
    # simple augmentation
    transform_train = transforms.Compose([
        transforms.RandomResizedCrop(args.input_size, scale=(0.2, 1.0), interpolation=3),  # 3 is bicubic
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    return transform_train

def minimum_transform():
    transform_train = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    return transform_train

def build_dataset(split, args, include_path=False, include_origin=False):
    return TinyImagenet(transform=simple_transform(args),
                        split=split,
                        subset=args.data_subset,
                        group=args.data_group,
                        include_path=include_path,
                        include_origin=include_origin)

class TinyImagenet(Dataset):
    def __init__(self, transform=None, split='train', subset=1.0, group=1,
                include_path=False, include_origin=False) -> None:
        super().__init__()
        self.split = split
        self.transform = transform
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
        if self.split == 'train' or self.split == 'val':
            img_path = os.path.join(os.path.join(self.path, row["class"]), row["id"])
            label = row["label"]
        else:
            img_path = os.path.join(self.path, row["id"])
        image = PIL.Image.open(img_path)
        image = image.convert("RGB")
        return self._get_custom_item(image, label, img_path)
    
    def _get_custom_item(self, image, label, img_path):
        basic_transform = minimum_transform()
        if self.transform:
            transformed_image = self.transform(image)
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
            return config("TINYIMAGENET_TRAIN_PATH")
        elif self.split == 'val':
            return config("TINYIMAGENET_VAL_PATH")
        else:
            return config("TINYIMAGENET_TEST_PATH")
    
    def _get_md(self):
        if self.split == 'train':
            df = pd.read_csv(config("TINYIMAGENET_TRAIN_META_PATH"))
            md = df.loc[df["group"] == self.group]
        elif self.split == 'val':
            md = pd.read_csv(config("TINYIMAGENET_VAL_META_PATH"))
        else:
            md = pd.read_csv(config("TINYIMAGENET_TEST_META_PATH"))
        assert len(md) > 0
        return md