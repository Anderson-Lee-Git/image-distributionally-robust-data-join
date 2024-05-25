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
from engine import train_one_epoch, evaluate
from utils.logs import log_stats, count_parameters
from dataset.datasets import build_dataset

def get_args_parser():
    parser = argparse.ArgumentParser('Convolution Autoencoder training', add_help=False)
    parser.add_argument('--batch_size', default=64, type=int,
                        help='Batch size per GPU (effective batch size is batch_size * accum_iter * # gpus')
    parser.add_argument('--epochs', default=80, type=int)

    # Optimizer parameters
    parser.add_argument('--weight_decay', type=float, default=0.05,
                        help='weight decay (default: 0.05)')
    parser.add_argument('--lr', type=float, default=None, metavar='LR',
                        help='learning rate (absolute lr)')
    parser.add_argument('--exp_lr_gamma', type=float, default=1.0,
                        help='gamma of exponential lr scheduler')
    
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
    parser.add_argument('--evaluate', action='store_true', default=False)
    parser.add_argument('--evaluate_ckpt', default='', type=str,
                        help="checkpoint path of model to evaluate")
    return parser

def main(args):
    print('job dir: {}'.format(os.path.dirname(os.path.realpath(__file__))))
    print("{}".format(args).replace(', ', ',\n'))
    # initilize wandb
    if args.use_wandb:
        wandb.init(project=args.project_name)
        args.output_dir = os.path.join(args.output_dir, wandb.run.id)
        args.log_dir = os.path.join(args.log_dir, wandb.run.id)
    device = torch.device(args.device)

    # fix seed for reproducibility
    seed = args.seed
    torch.manual_seed(seed)
    np.random.seed(seed)

    # set up dataset and data loaders
    print("Set up dataset and dataloader")
    dataset_train = build_dataset(args=args, split="train")
    dataset_val = build_dataset(args=args, split="val")
    data_loader_train = DataLoader(dataset_train, 
                                   shuffle=True,
                                   batch_size=args.batch_size,
                                   num_workers=args.num_workers,
                                   pin_memory=True)
    data_loader_val = DataLoader(dataset_val,
                                 batch_size=args.batch_size,
                                 num_workers=args.num_workers,
                                 pin_memory=True)
    # set up log writer
    if args.log_dir is not None:
        os.makedirs(args.log_dir, exist_ok=True)
        log_writer = SummaryWriter(log_dir=args.log_dir)
    # initialize model
    print("Load model")
    autoencoder = build_autoencoder(args).to(device)
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(autoencoder.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    lr_scheduler = ExponentialLR(optimizer, gamma=args.exp_lr_gamma)
    # information
    print(f"Number of parameters: {count_parameters(autoencoder)}")
    print(f"Number of training images: {len(dataset_train)}")
    print(f"Number of validation iamges: {len(dataset_val)}")
    # train loop
    print(f"Start training for {args.epochs} epochs")
    for epoch in range(args.epochs):
        visualize = epoch % 20 == 0 or epoch == args.epochs - 1
        train_loss = train_one_epoch(autoencoder=autoencoder,
                                     data_loader=data_loader_train,
                                     optimizer=optimizer,
                                     criterion=criterion,
                                     device=device,
                                     visualize=visualize,
                                     args=args)
        val_loss = evaluate(autoencoder=autoencoder,
                            data_loader=data_loader_val,
                            criterion=criterion,
                            device=device,
                            visualize=visualize,
                            args=args)
        # save model
        if args.output_dir and (epoch % 20 == 0 or epoch + 1 == args.epochs):
            to_save = {
                'autoencoder': autoencoder.state_dict(),
                'optimizer': optimizer.state_dict(),
                'epoch': epoch,
                'args': args
            }
            path = os.path.join(args.output_dir, f"checkpoint-{epoch}.pth")
            torch.save(to_save, path)
        # log stats
        log_stats(stats={"train_loss": train_loss, 
                         "val_loss": val_loss, 
                         "lr": lr_scheduler.get_last_lr()[0], 
                         "epoch": epoch},
                  log_writer=log_writer,
                  epoch=epoch,
                  args=args)
        lr_scheduler.step()

if __name__ == "__main__":
    args = get_args_parser()
    args = args.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
    if args.use_wandb:
        wandb.finish()
        os.system("wandb artifact cache cleanup 5G")