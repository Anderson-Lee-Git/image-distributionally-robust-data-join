import os
import sys
import random

from tqdm import tqdm
from sklearn import datasets
from torchvision import transforms
import torch
from torch import nn as nn
from torch.optim import AdamW
import numpy as np
from decouple import Config, RepositoryEnv
config = Config(RepositoryEnv(".env"))
sys.path.insert(0, config("REPO_ROOT"))

import MNISTConvNet


def mnist_transform():
    transform = transforms.Compose([
        transforms.ToTensor()
    ])
    return transform


def main():
    digits = datasets.load_digits()
    n = len(digits.data)
    shuffle_indices = list(range(n))
    random.shuffle(shuffle_indices)
    train_indices = shuffle_indices[:1400]
    validation_indices = shuffle_indices[1400:1550]
    test_indices = shuffle_indices[1550:]
    print(n)
    data_train, data_validation, data_test = digits.data[train_indices], digits.data[validation_indices], digits.data[test_indices]
    target_train, target_validation, target_test = digits.target[train_indices], digits.target[validation_indices], digits.target[test_indices]
    transform = mnist_transform()

    model = MNISTConvNet()
    optimizer = AdamW(model.parameters(), lr=1e-4)
    criterion = nn.CrossEntropyLoss()
    batch_size = 16
    loop = tqdm(total=len(data_train) // batch_size + 1, leave=True, pos=0)
    for i in range(0, len(data_train), batch_size):
        if i + batch_size < len(data_train):
            x = data_train[i:i+batch_size]
            y = target_train[i:i+batch_size]
        else:
            x = data_train[i:]
            y = target_train[i:]
        x = x.reshape(-1, 8, 8)
        x = transform(x)
        y_hat = model(x)
        loss = criterion(y, y_hat)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        loop.display(f"loss={loss.item()}")

        

if __name__ == "__main__":
    main()
