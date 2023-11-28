from typing import Iterable
import os
import sys

import torch
import wandb
from decouple import Config, RepositoryEnv
config = Config(RepositoryEnv(".env"))
sys.path.insert(0, config("REPO_ROOT"))

from utils.visualization import visualize_image

def train_one_epoch(encoder: torch.nn.Module,
                    decoder: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    criterion: torch.nn.MSELoss,
                    device: torch.device,
                    visualize: bool = False,
                    args=None):
    encoder.train()
    decoder.train()
    optimizer.zero_grad()
    loss = 0
    batch_cnt = 0
    for step, (samples, _, original_images) in enumerate(data_loader):
        samples = samples.to(device, non_blocking=True)
        original_images = original_images.to(device, non_blocking=True)
        latent = encoder(samples)
        reconstructed_images = decoder(latent)
        batch_loss = criterion(reconstructed_images, original_images)
        batch_loss.backward()
        optimizer.step()
        loss += batch_loss.item()
        batch_cnt += 1
    # visualize the last batch randomly
    p = torch.rand(1)
    if visualize and p < 0.02:
        if not os.path.exists(os.path.join(args.log_dir, "examples")):
            os.makedirs(os.path.join(args.log_dir, "examples"))
        visualize_image(samples[0], os.path.join(args.log_dir, "examples/sample_image.png"))
        visualize_image(reconstructed_images[0], os.path.join(args.log_dir, "examples/sample_reconstruction.png"))
    return loss / batch_cnt

def evaluate(encoder: torch.nn.Module,
            decoder: torch.nn.Module,
            data_loader: Iterable,
            criterion: torch.nn.MSELoss,
            device: torch.device,
            visualize: bool = False,
            args=None):
    encoder.eval()
    decoder.eval()
    loss = 0
    batch_cnt = 0
    with torch.no_grad():
        for step, (samples, _, original_images) in enumerate(data_loader):
            samples = samples.to(device, non_blocking=True)
            original_images = original_images.to(device, non_blocking=True)
            latent = encoder(samples)
            reconstructed_images = decoder(latent)
            batch_loss = criterion(reconstructed_images, original_images)
            # sample a random image/reconstruction pair in each batch to visualize
            if visualize:
                idx = torch.randint(low=0, high=len(samples), size=(1,)).item()
                if not os.path.exists(os.path.join(args.log_dir, "examples")):
                    os.makedirs(os.path.join(args.log_dir, "examples"))
                visualize_image(samples[idx], os.path.join(args.log_dir, f"examples/sample_image_{step}.png"))
                visualize_image(reconstructed_images[idx], os.path.join(args.log_dir, f"examples/sample_reconstruction_{step}.png"))
                if args.use_wandb:
                    wandb.log({
                        f"sample_image_{step}": wandb.Image(os.path.join(args.log_dir, f"examples/sample_image_{step}.png")),
                        f"sample_reconstruction_{step}": wandb.Image(os.path.join(args.log_dir, f"examples/sample_reconstruction_{step}.png"))
                    })
        loss += batch_loss.item()
        batch_cnt += 1
    return loss / batch_cnt

