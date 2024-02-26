import os
import sys
import argparse
from pathlib import Path
from torchvision import transforms
from tqdm import tqdm
import pandas as pd
import numpy as np
from transformers import AutoImageProcessor, ViTMAEForPreTraining, ViTMAEModel
import torch
from torch.utils.data import DataLoader
from torch.optim import Adam
import pickle
import matplotlib.pyplot as plt
from decouple import Config, RepositoryEnv
config = Config(RepositoryEnv(".env"))
sys.path.insert(0, config("REPO_ROOT"))

from dataset import CIFAR100, collate_fn

def get_latent_path(args):
    if args.unbalanced:
        return os.path.join(config("DATASET_ROOT"), config("CIFAR100_TRAIN_UNBALANCED_LATENT_PATH"))
    else:
        return os.path.join(config("DATASET_ROOT"), config("CIFAR100_TRAIN_LATENT_PATH"))

def get_args_parser():
    parser = argparse.ArgumentParser('MAE inference', add_help=False)
    parser.add_argument('--batch_size', default=64, type=int,
                        help='Batch size per GPU (effective batch size is batch_size * accum_iter * # gpus')
    # Dataset parameters
    parser.add_argument('--output_dir', default='/gscratch/jamiemmt/andersonlee/image-distributionally-robust-data-join/logs/srun',
                        help='path where to save, empty for no saving')
    parser.add_argument('--log_dir', default='/gscratch/jamiemmt/andersonlee/image-distributionally-robust-data-join/logs/srun',
                        help='path where to tensorboard log')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--unbalanced', action='store_true', default=False)
    parser.add_argument('--num_workers', default=5, type=int)
    parser.add_argument('--generate', action='store_true', default=False)
    parser.add_argument('--pretrained_path', default=None, type=str)
    return parser

@torch.no_grad()
def generate(dataloader: DataLoader,
             model: ViTMAEModel,
             image_processor):
    latent_dir = get_latent_path(args)
    print(f"output latent directory: {latent_dir}")
    latents = {}
    output_path = os.path.join(latent_dir, "mae_latents.pickle")
    f = open(output_path, "wb")
    for samples in tqdm(dataloader):
        original_images = samples["original_image"]
        B = len(original_images)
        inputs = image_processor(images=original_images, do_rescale=False, return_tensors="pt").to(model.device)
        out = model(**inputs)
        hidden_state = out.last_hidden_state.view(B, -1).cpu().numpy()
        paths = samples["path"]
        for i, path in enumerate(paths):
            image_id = path.split("/")[-1]
            latents[image_id] = hidden_state[i]
    print(hidden_state.shape)
    pickle.dump(latents, f)
    # for samples in tqdm(dataloader):
    #     dst_dirs = []
    #     original_images = samples["original_image"]
    #     labels = samples["label"]
    #     paths = samples["path"]
    #     B = original_images.size()[0]
    #     for i, path in enumerate(paths):
    #         image_id = path.split("/")[-1].split(".")[0] + ".npy"
    #         class_path = os.path.join(latent_dir, str(labels[i].item()))
    #         if not os.path.exists(class_path):
    #             os.mkdir(class_path)
    #         dst_dir = os.path.join(class_path, image_id)
    #         dst_dirs.append(dst_dir)
    #     inputs = image_processor(images=original_images, do_rescale=False, return_tensors="pt").to(model.device)
    #     latents = model(**inputs).last_hidden_state
    #     latents = latents.view(B, -1)
    #     latents = latents.cpu().numpy()
    #     for i, dst_dir in enumerate(dst_dirs):
    #         np.save(dst_dir, latents[i])
    # print(latents.shape)
    
def main(args):
    device = "cuda"
    cache_path = "/gscratch/jamiemmt/andersonlee/image-distributionally-robust-data-join/logs/cache"
    no_transform = transforms.Compose([
        transforms.ToTensor()
    ])
    dataset = CIFAR100(train_transform=no_transform,
                        minimum_transform=no_transform,
                        val_transform=no_transform,
                        split="train",
                        subset=1.0,
                        group=0,
                        unbalanced=False,
                        include_path=True)
    dataset.collate_fn = collate_fn
    dataloader = DataLoader(dataset=dataset,
                            batch_size=args.batch_size,
                            shuffle=True,
                            pin_memory=True,
                            num_workers=args.num_workers,
                            collate_fn=dataset.collate_fn)
    image_processor = AutoImageProcessor.from_pretrained("facebook/vit-mae-base", cache_dir=cache_path)
    if args.generate:
        model = ViTMAEForPreTraining.from_pretrained(args.pretrained_path, cache_dir=cache_path).to(device)
        generate(dataloader, model.vit, image_processor)
    else:
        model = ViTMAEForPreTraining.from_pretrained("facebook/vit-mae-base", cache_dir=cache_path).to(device)
        optimizer = Adam(model.parameters(), lr=1e-5)
        losses = []
        num_epochs = 3
        for epoch in range(num_epochs):
            print(f"Epoch: {epoch+1}")
            for samples in dataloader:
                loop = tqdm(total=len(dataloader), leave=False, position=0)
                original_images = samples["original_image"]
                inputs = image_processor(images=original_images, do_rescale=False, return_tensors="pt").to(device)
                out = model(**inputs)
                loss = out.loss
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                loop.set_description(f"Loss: {loss.item():.4f}")
                loop.update(1)
                losses.append(loss.item())
        model.save_pretrained(f"{config('REPO_ROOT')}/mae/checkpoints/mae_ft_cifar100_trial_1", from_pt=True)
        plt.plot(losses)
        plt.savefig(f"{args.output_dir}/mae_loss.png")

if __name__ == "__main__":
    args = get_args_parser()
    args = args.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)