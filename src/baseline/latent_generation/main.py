import argparse
from pathlib import Path
import os
import sys
import pickle

import wandb
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import ExponentialLR
from tqdm import tqdm
from decouple import Config, RepositoryEnv
config = Config(RepositoryEnv(".env"))
sys.path.insert(0, config("REPO_ROOT"))

from dataset.datasets import build_dataset
from models import build_resnet
from engine import generate

def get_args_parser():
    parser = argparse.ArgumentParser('Baseline inference', add_help=False)
    parser.add_argument('--batch_size', default=64, type=int,
                        help='Batch size per GPU (effective batch size is batch_size * accum_iter * # gpus')
    
    # Model parameters
    parser.add_argument('--model', type=str, default="resnet50",
                        help='which model to use')

    # Dataset parameters
    parser.add_argument('--input_size', type=int, default=32)
    parser.add_argument('--output_dir', default='/gscratch/jamiemmt/andersonlee/image-distributionally-robust-data-join/logs/srun',
                        help='path where to save, empty for no saving')
    parser.add_argument('--log_dir', default='/gscratch/jamiemmt/andersonlee/image-distributionally-robust-data-join/logs/srun',
                        help='path where to tensorboard log')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--resume', default='',
                        help='resume from checkpoint')
    parser.add_argument('--dataset', default="cifar100", type=str, help='dataset option')
    parser.add_argument('--data_group', default=0, type=int)
    parser.add_argument('--unbalanced', action='store_true', default=False)
    parser.add_argument('--num_classes', default=None, type=int)

    parser.add_argument('--num_workers', default=5, type=int)
    parser.add_argument('--data_subset', default=1.0, type=float,
                        help='subset of data to use')
    parser.add_argument('--latent_path', default=None, type=str)
    
    # evaluation
    parser.add_argument('--ckpt', default='', type=str,
                        help="checkpoint path of model to evaluate")
    return parser

def main(args):
    if args.latent_path is None:
        raise ValueError("latent path required")
    device = torch.device(args.device)
    # fix seed for reproducibility
    seed = args.seed
    torch.manual_seed(seed)
    np.random.seed(seed)

    dataset_train = build_dataset(args=args, split="train", include_path=True)
    dataloader = DataLoader(dataset=dataset_train,
                            batch_size=args.batch_size,
                            num_workers=args.num_workers,
                            pin_memory=True)
    # load model from ckpt
    model = build_resnet(num_classes=args.num_classes, pretrained=True, args=args)
    if args.ckpt != '':
        state_dict = torch.load(args.ckpt)
        model.load_state_dict(state_dict["model"])
    model.encoder.fc = nn.Identity()
    model.avpool = nn.Identity()
    model.fc = nn.Identity()
    model.to(device)

    output_path = args.latent_path
    generate(dataloader=dataloader,
             model=model,
             output_path=output_path)

if __name__ == "__main__":
    args = get_args_parser()
    args = args.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)