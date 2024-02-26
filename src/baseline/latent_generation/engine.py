import pickle

import torch
from torch.utils.data import DataLoader
import torch.nn as nn
from tqdm import tqdm

@torch.no_grad()
def generate(dataloader: DataLoader,
             model: nn.Module,
             output_path: str):
    latents = {}
    f = open(output_path, "wb")
    for samples in tqdm(dataloader):
        original_images = samples["original_image"].to("cuda")
        B = len(original_images)
        hidden_state = model(original_images).view(B, -1).cpu().numpy()
        paths = samples["path"]
        for i, path in enumerate(paths):
            image_id = path.split("/")[-1]
            latents[image_id] = hidden_state[i]
    print(hidden_state.shape)
    pickle.dump(latents, f)