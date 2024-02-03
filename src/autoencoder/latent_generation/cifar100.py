import argparse
from pathlib import Path
import os
import sys

import wandb
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import ExponentialLR
from tqdm import tqdm
from decouple import Config, RepositoryEnv
config = Config(RepositoryEnv(".env"))
sys.path.insert(0, config("REPO_ROOT"))

from models import build_autoencoder
from utils.logs import log_stats, count_parameters
from dataset.datasets import build_dataset

def get_args_parser():
    parser = argparse.ArgumentParser('Convolution Autoencoder inference', add_help=False)
    parser.add_argument('--batch_size', default=64, type=int,
                        help='Batch size per GPU (effective batch size is batch_size * accum_iter * # gpus')
    
    # Model parameters
    parser.add_argument('--model', type=str, default="basic_conv_256",
                        help='which model to use')

    # Dataset parameters
    parser.add_argument('--input_size', type=int, default=32)
    parser.add_argument('--output_dir', default='./output_dir',
                        help='path where to save, empty for no saving')
    parser.add_argument('--log_dir', default='./output_dir',
                        help='path where to tensorboard log')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--resume', default='',
                        help='resume from checkpoint')
    parser.add_argument('--dataset', default=None, type=str, help='dataset option')
    parser.add_argument('--data_group', default=0, type=int)

    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--num_workers', default=5, type=int)
    parser.add_argument('--data_subset', default=1.0, type=float,
                        help='subset of data to use')
    # misc
    parser.add_argument('--use_wandb', action='store_true', default=False)
    parser.add_argument('--project_name', default='', type=str,
                        help='wandb project name')
    
    # evaluation
    parser.add_argument('--ckpt', default='', type=str,
                        help="checkpoint path of model to evaluate")
    return parser

@torch.no_grad()
def main(args):
    device = torch.device(args.device)
    # fix seed for reproducibility
    seed = args.seed
    torch.manual_seed(seed)
    np.random.seed(seed)

    args.dataset = "cifar100"
    dataset_train = build_dataset(args=args, split="train", include_path=True)
    dataloader = DataLoader(dataset=dataset_train,
                            batch_size=args.batch_size,
                            num_workers=args.num_workers,
                            pin_memory=True)
    # load model from ckpt
    model = build_autoencoder(args)
    state_dict = torch.load(args.ckpt, map_location=device)
    model.load_state_dict(state_dict["autoencoder"])
    model.to(device)

    latent_dir = config("CIFAR100_TRAIN_LATENT_PATH")

    for images, original_images, labels, paths in tqdm(dataloader):
        dst_dirs = []
        for i, path in enumerate(paths):
            image_id = path.split("/")[-1].split(".")[0] + ".npy"
            class_path = os.path.join(latent_dir, str(labels[i].item()))
            if not os.path.exists(class_path):
                os.mkdir(class_path)
            dst_dir = os.path.join(class_path, image_id)
            dst_dirs.append(dst_dir)
        images = images.to(device)
        latents = model.encoder(images)
        latents = latents.cpu().numpy()
        for i, dst_dir in enumerate(dst_dirs):
            np.save(dst_dir, latents[i])

if __name__ == "__main__":
    args = get_args_parser()
    args = args.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)