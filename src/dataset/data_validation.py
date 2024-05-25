import os
import pickle
import sys
sys.path.insert(0, ".")

import PIL
import torch
from torch.utils.data import DataLoader
import pandas as pd
from torchvision import transforms
from cifar_100_pairs import CIFAR100Pairs
from cifar_100 import CIFAR100
from celebA import CelebA
from decouple import Config, RepositoryEnv
config = Config(RepositoryEnv(".env"))

# Actual collate fn in datasets.py
def collate_fn(batched_samples):
    assert len(batched_samples) > 0
    batch = {}
    batch["image"] = torch.stack([sample["image"] for sample in batched_samples], dim=0)
    batch["label"] = torch.stack([sample["label"] for sample in batched_samples], dim=0)
    if "original_image" in batched_samples[0]:
        batch["original_image"] = torch.stack([sample["original_image"] for sample in batched_samples], dim=0)
    if "aux" in batched_samples[0]:
        batch["aux"] = torch.stack([sample["aux"] for sample in batched_samples], dim=0)
    for k in batched_samples[0]:
        if k not in batch:
            batch[k] = [sample[k] for sample in batched_samples]
    return batch

transform = transforms.Compose([
            transforms.Resize(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5071, 0.4867, 0.4408], std=[0.2675, 0.2565, 0.2761])])
dataset = CelebA(train_transform=transform,
                   minimum_transform=transform,
                   val_transform=transform,
                    subset=1.0)
dataloader = DataLoader(dataset, batch_size=20, shuffle=True, collate_fn=collate_fn)
for i, sample in enumerate(dataloader):
    images = sample["image"]
    labels = sample["label"]
    aux = sample["aux"]
    print(labels)
    print(aux)
    print(images.shape)
    print(type(images))
    print(labels.shape)
    print(type(labels))
    print(aux.shape)
    print(type(aux))
    break
    