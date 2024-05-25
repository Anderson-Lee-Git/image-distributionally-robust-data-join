import argparse
from pathlib import Path
import os
import sys

import wandb
import numpy as np
import torch
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import ExponentialLR
from tqdm import tqdm
from decouple import Config, RepositoryEnv
config = Config(RepositoryEnv(".env"))
sys.path.insert(0, config("REPO_ROOT"))

from models import Adversarial
from engine import train_one_epoch, evaluate
from utils.logs import log_stats, count_parameters
from utils.args_utils import get_wandb_args
from dataset.datasets import build_dataset, GroupCollateFnClass

def get_args_parser():
    parser = argparse.ArgumentParser('baseline training', add_help=False)
    parser.add_argument('--batch_size', default=64, type=int,
                        help='Batch size per GPU (effective batch size is batch_size * accum_iter * # gpus')
    parser.add_argument('--epochs', default=80, type=int)

    # Model parameters
    parser.add_argument('--model', default='resnet50', type=str)
    parser.add_argument('--beta', type=float, default=0.1)

    # Optimizer parameters
    parser.add_argument('--weight_decay', type=float, default=0.00001,
                        help='weight decay (default: 0.05)')
    parser.add_argument('--lr', type=float, default=None, metavar='LR',
                        help='learning rate (absolute lr)')
    parser.add_argument('--exp_lr_gamma', type=float, default=1.0,
                        help='gamma of exponential lr scheduler')

    # Dataset parameters
    parser.add_argument('--input_size', type=int, default=64)
    parser.add_argument('--output_dir', default='/gscratch/jamiemmt/andersonlee/image-distributionally-robust-data-join/logs/srun',
                        help='path where to save, empty for no saving')
    parser.add_argument('--log_dir', default='/gscratch/jamiemmt/andersonlee/image-distributionally-robust-data-join/logs/srun',
                        help='path where to tensorboard log')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--resume', default='',
                        help='resume from checkpoint')
    parser.add_argument('--dataset', default='cifar100', type=str, help='dataset option')
    parser.add_argument('--num_classes', default=100, type=int)

    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--num_workers', default=5, type=int)
    parser.add_argument('--data_subset', default=1.0, type=float,
                        help='subset of data to use')
    parser.add_argument('--data_group', default=1, type=int)
    parser.add_argument('--unbalanced', action='store_true', default=False)

    # misc
    parser.add_argument('--use_wandb', action='store_true', default=False)
    parser.add_argument('--wandb_cont_sweep', action='store_true', default=False)
    parser.add_argument('--project_name', default='', type=str,
                        help='wandb project name')
    
    # evaluation
    parser.add_argument('--evaluate', action='store_true', default=False)
    parser.add_argument('--evaluate_ckpt', default='', type=str,
                        help="checkpoint path of model to evaluate")
    return parser

def main():
    args = get_args_parser()
    args = args.parse_args()

    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    # initilize wandb
    if args.use_wandb:
        run = wandb.init(project=args.project_name, dir=config("REPO_ROOT"))
        args.output_dir = os.path.join(args.output_dir, wandb.run.name)
        args.log_dir = os.path.join(args.log_dir, wandb.run.name)
        args = get_wandb_args(run, args)
    
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    print('job dir: {}'.format(os.path.dirname(os.path.realpath(__file__))))
    print("{}".format(args).replace(', ', ',\n'))
    current_working_directory = os.getcwd()
    print(f"current directory: {current_working_directory}")

    device = torch.device(args.device)

    # fix seed for reproducibility
    seed = args.seed
    torch.manual_seed(seed)
    np.random.seed(seed)

    # set up dataset and data loaders
    print("Set up dataset and dataloader")
    args.data_group = 1
    target_dataset_train = build_dataset(split='train', args=args)
    args.data_group = 2
    attr_dataset_train = build_dataset(split='train', args=args)
    dataset_val = build_dataset(split="val", args=args)
    print(target_dataset_train)
    print(attr_dataset_train)
    print(dataset_val)
    target_dataloader_train = DataLoader(target_dataset_train, 
                                         shuffle=True,
                                         batch_size=args.batch_size,
                                         num_workers=args.num_workers,
                                         collate_fn=target_dataset_train.collate_fn,
                                         pin_memory=True)
    attr_dataloader_train = DataLoader(attr_dataset_train, 
                                       shuffle=True,
                                       batch_size=args.batch_size,
                                       num_workers=args.num_workers,
                                       collate_fn=attr_dataset_train.collate_fn,
                                       pin_memory=True)
    dataloader_val = DataLoader(dataset_val,
                                 batch_size=args.batch_size,
                                 num_workers=args.num_workers,
                                 pin_memory=True)
    # set up log writer
    if args.log_dir is not None:
        os.makedirs(args.log_dir, exist_ok=True)
        log_writer = SummaryWriter(log_dir=args.log_dir)
    # initialize model
    model = Adversarial(embed_dim=2048, num_target_classes=2, num_attr_classes=2, beta=args.beta, args=args)
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    lr_scheduler = ExponentialLR(optimizer, gamma=args.exp_lr_gamma)
    print(f"Number of params: {count_parameters(model)}")
    # groups
    groups = np.transpose([np.tile(range(2), args.num_classes),
                           np.repeat(range(args.num_classes), 2)])
    # train loop
    print(f"Start training for {args.epochs} epochs")
    criterion = CrossEntropyLoss()
    best_acc = 0
    target_flag = True
    for epoch in range(args.epochs):
        print(f"Epoch {epoch+1}")
        option = "target" if target_flag else "attr"
        if option == "target":
            train_loss, train_acc = train_one_epoch(model=model, 
                                                    dataloader=target_dataloader_train,
                                                    optimizer=optimizer,
                                                    device=device,
                                                    option=option,
                                                    args=args)
        else:
            train_loss, train_acc = train_one_epoch(model=model, 
                                                    dataloader=attr_dataloader_train,
                                                    optimizer=optimizer,
                                                    device=device,
                                                    option=option,
                                                    args=args)
        for group in groups:
            GroupCollateFnClass.group = (torch.tensor(group[0]), torch.tensor(group[1]))
            dataset_val.collate_fn = GroupCollateFnClass.group_filter_collate_fn
            dataloader_val = DataLoader(dataset_val,
                                        batch_size=args.batch_size,
                                        num_workers=args.num_workers,
                                        collate_fn=dataset_val.collate_fn,
                                        pin_memory=True)
            val_loss, val_acc = evaluate(model=model,
                                        dataloader=dataloader_val,
                                        criterion=criterion,
                                        device=device,
                                        args=args)
            log_stats(stats={f"val_loss ({group[0]}, {group[1]})": val_loss,
                             f"val_acc ({group[0]}, {group[1]})": val_acc},
                    log_writer=log_writer,
                    epoch=epoch,
                    args=args,
                    commit=False)
        # save model
        if args.output_dir and (epoch % 20 == 0 or epoch + 1 == args.epochs):
            to_save = {
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'epoch': epoch,
                'args': args
            }
            path = os.path.join(args.output_dir, f"checkpoint-{epoch}.pth")
            torch.save(to_save, path)
        if val_acc > best_acc:
            to_save = {
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'epoch': epoch,
                'args': args
            }
            path = os.path.join(args.output_dir, f"checkpoint-best.pth")
            torch.save(to_save, path)
        # log stats
        log_stats(stats={f"train_loss ({option})": train_loss,
                         f"train_acc ({option})": train_acc,
                        #  "val_loss": val_loss,
                        #  "val_acc": val_acc,
                         "lr": lr_scheduler.get_last_lr()[0],
                         "epoch": epoch},
                  log_writer=log_writer,
                  epoch=epoch,
                  args=args)
        lr_scheduler.step()
        if epoch % 2 == 0:
            target_flag = not target_flag

if __name__ == "__main__":
    args = get_args_parser()
    args = args.parse_args()
    
    if args.wandb_cont_sweep:
        sweep_configuration = {
            "method": "random",
            "metric": {"goal": "maximize", "name": "val_acc"},
            "parameters":
            {
                "lr": {"max": 1e-4, "min": 1e-5},
                "weight_decay": {"max": 1e-4, "min": 1e-5},
                "exp_lr_gamma": {"values": [0.99]},
            }
        }
        sweep_id = wandb.sweep(sweep=sweep_configuration, project=args.project_name)
        wandb.agent(sweep_id, function=main, count=5)
    else:
        main()