import os
import pickle
import sys
sys.path.insert(0, ".")

import PIL
from torch.utils.data import DataLoader, RandomSampler
import pandas as pd
from torchvision import transforms
from decouple import Config, RepositoryEnv
config = Config(RepositoryEnv(".env"))

from aux_cifar_100 import AuxCIFAR100

dataset = AuxCIFAR100(train_transform=transforms.Compose([
                                    transforms.ToTensor(),
                                    transforms.Normalize(mean=[0.5071, 0.4867, 0.4408], std=[0.2675, 0.2565, 0.2761])]),
                        val_transform=transforms.Compose([
                                        transforms.ToTensor(),
                                        transforms.Normalize(mean=[0.5071, 0.4867, 0.4408], std=[0.2675, 0.2565, 0.2761])]),
                        split="train")
sampler_train = RandomSampler(dataset)
dataloader = DataLoader(dataset, batch_size=64, sampler=sampler_train)
for i, image in enumerate(dataloader):
    print(type(image))
    print(image.shape)