import numpy as np
import os
import sys
import time
import json
from PIL import Image
from torchvision import transforms
from tqdm import tqdm
import pandas as pd
from transformers import AutoImageProcessor, ViTMAEModel
from decouple import Config, RepositoryEnv
config = Config(RepositoryEnv(".env"))
sys.path.insert(0, config("REPO_ROOT"))

from dataset import CIFAR100, collate_fn

md = pd.read_csv(config("CIFAR100_TRAIN_META_PATH"))
path_root = config("CIFAR100_TRAIN_LATENT_PATH")

device = "cuda"
cache_path = "/gscratch/jamiemmt/andersonlee/image-distributionally-robust-data-join/logs/cache"
no_transform = transforms.Compose([
    transforms.ToTensor()
])
dataset = CIFAR100(train_transform=no_transform,
                   minimum_transform=no_transform,
                   val_transform=no_transform,
                   split="train",
                   subset=0.1,
                   group=0,
                   unbalanced=False,
                   include_path=False)

sample = dataset[0]
image = sample["image"]
image_processor = AutoImageProcessor.from_pretrained("facebook/vit-mae-base", cache_dir=cache_path)
model = ViTMAEModel.from_pretrained("facebook/vit-mae-base", cache_dir=cache_path).to(device)

inputs = image_processor(images=image, do_rescale=False, return_tensors="pt").to(device)
print(inputs["pixel_values"].size())
outputs = model(**inputs)
last_hidden_states = outputs.last_hidden_state
print(last_hidden_states.size())