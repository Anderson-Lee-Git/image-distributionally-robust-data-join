# https://github.com/emma-mens/elk-recognition/blob/main/src/multimodal_species/datasets/birds_dataset.py#L498
import torch
from torch.utils.data import Dataset
import numpy as np
import h5py
from decouple import Config, RepositoryEnv
config = Config(RepositoryEnv(".env"))

class ImagenetDataset(Dataset):
    def __init__(self, transform=None, split='train', size=256, crop=224, subset=1.0):
        # get length of total data for train and for val (test)
        # shuffle arange(train length) and pick last 20% for validation
        # index into imagenet h5py
        self.split = split
        self.transform = transform
        # self.classifier_transform = CLASSIFIER_TRANSFORM
        self.path = config("IMAGENET-PATH")
        self.h5_split = 'valid' if split == 'test' else 'train'
        self.val_p = 0.2
        with h5py.File(self.path, 'r') as hf:
            labels = np.array(hf[f"/{self.h5_split}/labels"][:]).reshape(-1)

        idx = np.arange(len(labels))
        # make train/val split
        train, val = idx, idx
        if split != 'test':
            train, val = self._train_val_split(idx, labels, self.val_p)
        if split == 'train':
            self.idx = train
        elif split == 'val':
            self.idx = val
        else:
            self.idx = idx
        # take subset of dataset if needed
        if subset < 1.0 and split == 'train':
            self.idx = np.random.choice(self.idx, int(subset*len(self.idx)))

    def __len__(self):
        return len(self.idx)

    def __del__(self):
        if hasattr(self, 'hdf5_path'):
            self.hdf5_path.close()

    def open_hdf5(self):
        self.hdf5_path = h5py.File(self.path, 'r')
        self.labels = self.hdf5_path[f'/{self.h5_split}/labels']
        self.imgs = self.hdf5_path[f'/{self.h5_split}/images']
    
    def _train_val_split(self, idx, labels, val_p=0.2):
        train = []
        val = []
        for c in np.unique(labels):
            c = idx[labels == c]
            # shuffle
            np.random.shuffle(c)
            n_val = int(val_p*len(c))
            n_train = len(c) - n_val
            train.extend(idx[c[:n_train]])
            val.extend(idx[c[-n_val:]])
        return train, val
    
    def __getitem__(self, idx):
        
        if not hasattr(self, f'hdf5_path'):
            # https://github.com/pytorch/pytorch/issues/11929#issuecomment-649760983
            self.open_hdf5()
        image, label = self.imgs[idx], self.labels[idx][0]
        image = (image - image.min())/(image.max() - image.min()) * 255
        if self.transform:
            image = self.transform(image)
        image = image.astype(np.uint8)
        return image, torch.tensor(label).long()