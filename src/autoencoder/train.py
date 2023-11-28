import argparse
from pathlib import Path
import os
import sys

import wandb
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from decouple import Config, RepositoryEnv
config = Config(RepositoryEnv(".env"))
sys.path.insert(0, config("REPO_ROOT"))

from model import ConvDecoder, ConvEncoder
from engine import train_one_epoch, evaluate
from utils.logs import log_stats
from dataset.tiny_imagenet import build_dataset

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
    # parser.add_argument('--blr', type=float, default=1e-3, metavar='LR',
    #                     help='base learning rate: absolute_lr = base_lr * total_batch_size / 256')
    # parser.add_argument('--min_lr', type=float, default=0., metavar='LR',
    #                     help='lower lr bound for cyclic schedulers that hit 0')
    # parser.add_argument('--warmup_epochs', type=int, default=40, metavar='N',
    #                     help='epochs to warmup LR')

    # Dataset parameters
    parser.add_argument('--input_size', type=int, default=64)
    parser.add_argument('--output_dir', default='./output_dir',
                        help='path where to save, empty for no saving')
    parser.add_argument('--log_dir', default='./output_dir',
                        help='path where to tensorboard log')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--resume', default='',
                        help='resume from checkpoint')
    parser.add_argument('--dataset', default='tiny_imagenet', type=str, help='dataset option')
    parser.add_argument('--data_group', default=1, type=int, help='which data group')

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
        args.output_dir = os.path.join(args.output_dir, wandb.run.name)
        args.log_dir = os.path.join(args.log_dir, wandb.run.name)
    device = torch.device(args.device)

    # fix seed for reproducibility
    seed = args.seed
    torch.manual_seed(seed)
    np.random.seed(seed)

    # set up dataset and data loaders
    print("Set up dataset and dataloader")
    dataset_train = build_dataset(split="train", args=args, include_origin=True)
    dataset_val = build_dataset(split="val", args=args, include_origin=True)
    sampler_train = torch.utils.data.RandomSampler(dataset_train)
    data_loader_train = DataLoader(dataset_train, sampler=sampler_train,
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
    encoder = ConvEncoder().to(device)
    decoder = ConvDecoder().to(device)
    criterion = torch.nn.MSELoss()
    params = list(encoder.parameters()) + list(decoder.parameters())
    optimizer = torch.optim.AdamW(params, lr=args.lr)
    # train loop
    print(f"Start training for {args.epochs} epochs")
    for epoch in tqdm(range(args.epochs)):
        visualize = epoch % 20 == 0 or epoch == args.epochs - 1
        train_loss = train_one_epoch(encoder=encoder, decoder=decoder,
                                     data_loader=data_loader_train,
                                     optimizer=optimizer,
                                     criterion=criterion,
                                     device=device,
                                     visualize=visualize,
                                     args=args)
        val_loss = evaluate(encoder=encoder, decoder=decoder,
                            data_loader=data_loader_val,
                            criterion=criterion,
                            device=device,
                            visualize=visualize,
                            args=args)
        # save model
        if args.output_dir and (epoch % 20 == 0 or epoch + 1 == args.epochs):
            to_save = {
                'encoder': encoder.state_dict(),
                'decoder': decoder.state_dict(),
                'optimizer': optimizer.state_dict(),
                'epoch': epoch,
                'args': args
            }
            path = os.path.join(args.output_dir, f"checkpoint-{epoch}.pth")
            torch.save(to_save, path)
        # log stats
        log_stats(stats={"train_loss": train_loss, 
                         "val_loss": val_loss, 
                         "lr": args.lr, 
                         "epoch": epoch},
                  log_writer=log_writer,
                  epoch=epoch,
                  args=args)

if __name__ == "__main__":
    args = get_args_parser()
    args = args.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
    if args.use_wandb:
        wandb.finish()