import argparse
from argparse import Namespace
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

from models import DRDJVanilla
from engine import train_one_epoch, evaluate
from utils.logs import log_stats
from utils.args_utils import get_wandb_args
from dataset.datasets import build_dataset

def get_args_parser():
    parser = argparse.ArgumentParser('DRDJ vanilla optimization', add_help=False)
    parser.add_argument('--batch_size', default=64, type=int,
                        help='Batch size per GPU (effective batch size is batch_size * accum_iter * # gpus')
    parser.add_argument('--epochs', default=80, type=int)

    # Model parameters
    parser.add_argument('--r_a', default=1.65, type=float)
    parser.add_argument('--r_p', default=1.65, type=float)
    parser.add_argument('--lambda_1', default=0.0, type=float)
    parser.add_argument('--lambda_2', default=0.0, type=float)
    parser.add_argument('--lambda_3', default=0.0, type=float)
    parser.add_argument('--kappa_a', default=5, type=float)
    parser.add_argument('--kappa_p', default=5, type=float)
    parser.add_argument('--model', default='resnet50', type=str)
    parser.add_argument('--pretrained_path', default=None, type=str)
    parser.add_argument('--embed_dim', default=2048, type=int)
    parser.add_argument('--aux_embed_dim', default=32, type=int)
    parser.add_argument('--freeze_params', default=False, type=bool)

    # Optimizer parameters
    parser.add_argument('--weight_decay', type=float, default=0.05,
                        help='weight decay (default: 0.05)')
    parser.add_argument('--lr', type=float, default=None, metavar='LR',
                        help='learning rate (absolute lr)')
    parser.add_argument('--alpha_lr', type=float, default=None)
    parser.add_argument('--cls_lr', type=float, default=1e-4)
    parser.add_argument('--aux_lr', type=float, default=1e-4)
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
    parser.add_argument('--dataset', default='cifar100_pairs', type=str, help='dataset option')
    parser.add_argument('--num_classes', default=200, type=int)

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

def get_optimizer(model: DRDJVanilla, args):
    optimizer = torch.optim.Adam(model.encoder.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    optimizer.add_param_group({
        'params': model.aux_encoder.parameters(),
        'lr': args.aux_lr,
        "weight_decay": args.weight_decay
    })
    optimizer.add_param_group({
        'params': model.fc.parameters(),
        'lr': args.cls_lr,
        "weight_decay": args.weight_decay
    })
    optimizer.add_param_group({
        'params': model.alpha_a,
        'lr': args.alpha_lr
    })
    optimizer.add_param_group({
        'params': model.alpha_p,
        'lr': args.alpha_lr
    })
    return optimizer

def get_model(n_a, n_p, objective, args):
    model = DRDJVanilla(r_a=args.r_a, r_p=args.r_p,
                        kappa_a=args.kappa_a, kappa_p=args.kappa_p,
                        n_a=n_a, n_p=n_p,
                        lambda_1=args.lambda_1,
                        lambda_2=args.lambda_2,
                        lambda_3=args.lambda_3,
                        backbone=args.model,
                        num_classes=args.num_classes,
                        embed_dim=args.embed_dim,
                        aux_embed_dim=args.aux_embed_dim,
                        objective=objective,
                        args=args)
    return model

def main():
    args = get_args_parser()
    args = args.parse_args()

    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    
    # initilize wandb
    if args.use_wandb:
        run = wandb.init(project=args.project_name)
        args.output_dir = os.path.join(args.output_dir, wandb.run.name)
        args.log_dir = os.path.join(args.log_dir, wandb.run.name)
        if not os.path.exists(os.path.join(args.output_dir, "examples")):
            os.makedirs(os.path.join(args.output_dir, "examples"), exist_ok=True)
        print(f"wandb run name: {wandb.run.name}")
    
    if args.wandb_cont_sweep:
        # override the args parameters with wandb run's configuration
        args = get_wandb_args(run, args)
    
    print('job dir: {}'.format(os.path.dirname(os.path.realpath(__file__))))
    print("{}".format(args).replace(', ', ',\n'))
    
    device = torch.device(args.device)

    # fix seed for reproducibility
    seed = args.seed
    torch.manual_seed(seed)
    np.random.seed(seed)

    # set up dataset and data loaders
    print("Set up dataset and dataloader")
    dataset_train = build_dataset(args=args, split="train")
    dataset_val = build_dataset(args=args, split="val")
    print(f"Total number of pairs: {len(dataset_train)}")
    print(dataset_train)
    print(dataset_val)
    sampler_train = torch.utils.data.RandomSampler(dataset_train)
    data_loader_train = DataLoader(dataset_train, sampler=sampler_train,
                                   batch_size=args.batch_size,
                                   num_workers=args.num_workers,
                                   collate_fn=dataset_train.collate_fn,
                                   pin_memory=True)
    data_loader_val = DataLoader(dataset_val,
                                 batch_size=args.batch_size,
                                 num_workers=args.num_workers,
                                 collate_fn=dataset_val.collate_fn,
                                 pin_memory=True)
    # set up log writer
    if args.log_dir is not None:
        os.makedirs(args.log_dir, exist_ok=True)
        log_writer = SummaryWriter(log_dir=args.log_dir)
    # initialize model
    print(f"Load model: {args.model}")
    n_a, n_p = dataset_train.get_len_per_group()
    # two models for dual objectives
    model_P = get_model(n_a, n_p, objective="P", args=args).to(device)
    # model_A = get_model(n_a, n_p, objective="A", args=args).to(device)
    if args.freeze_params:
        print("Freeze encoder parameters")
        model_P.freeze_params()
        # model_A.freeze_params()
    total_params = sum(p.numel() for p in model_P.parameters() if p.requires_grad)
    print(f"Total trainable parameters for model_P: {total_params}")
    # total_params = sum(p.numel() for p in model_A.parameters() if p.requires_grad)
    # print(f"Total trainable parameters for model_A: {total_params}")
    optimizer_P = get_optimizer(model_P, args)
    # optimizer_A = get_optimizer(model_A, args)
    lr_scheduler_P = ExponentialLR(optimizer_P, gamma=args.exp_lr_gamma)
    # lr_scheduler_A = ExponentialLR(optimizer_A, gamma=args.exp_lr_gamma)

    epoch = 0
    latest_ckpt_path = os.path.join(args.output_dir, f"checkpoint-latest.pth")
    best_ckpt_path = os.path.join(args.output_dir, f"checkpoint-best.pth")
    best_val_acc = 0
    # load resumed jobs
    if args.use_wandb and wandb.run.resumed:
        ckpt = torch.load(wandb.restore(latest_ckpt_path))
        model_P.load_state_dict(ckpt["model_P"])
        # model_A.load_state_dict(ckpt["model_A"])
        optimizer_P.load_state_dict(ckpt["optimizer_P"])
        # optimizer_A.load_state_dict(ckpt["optimizer_A"])
        epoch = ckpt["epoch"]
    # train loop
    print(f"Start training for {args.epochs} epochs from epoch = {epoch}")
    # torch.autograd.detect_anomaly(True)
    while epoch < args.epochs:
        print(f"Epoch {epoch+1}")
        train_loss_P, train_acc_P = train_one_epoch(model=model_P, data_loader=data_loader_train,
                                                optimizer=optimizer_P,
                                                device=device,
                                                include_max_term=True,
                                                include_norm=True,
                                                args=args)
        # train_loss_A, train_acc_A = train_one_epoch(model=model_A, data_loader=data_loader_train,
        #                                         optimizer=optimizer_A,
        #                                         device=device,
        #                                         include_max_term=True,
        #                                         include_norm=True,
        #                                         args=args)
        val_loss_P, val_acc_P = evaluate(model=model_P, data_loader=data_loader_val,
                                        criterion=torch.nn.CrossEntropyLoss(),
                                        device=device,
                                        args=args)
        # val_loss_A, val_acc_A = evaluate(model=model_A, data_loader=data_loader_val,
        #                                 criterion=torch.nn.CrossEntropyLoss(),
        #                                 device=device,
        #                                 args=args)

        # save model
        # TODO: Change to save on some epochs
        # if val_acc_P > val_acc_A:
        #     objective = "P"
        # else:
        #     objective = "A"
        to_save = {
            'model_P': model_P.state_dict(),
            'optimizer_P': optimizer_P.state_dict(),
            # 'model_A': model_A.state_dict(),
            # 'optimizer_A': optimizer_A.state_dict(),
            'epoch': epoch,
            'args': args,
            # 'objective': objective
        }
        if args.output_dir and (epoch % 3 == 0 or epoch + 1 == args.epochs):
            path = os.path.join(args.output_dir, f"checkpoint-{epoch}.pth")
            torch.save(to_save, path)
        # save latest
        torch.save(to_save, latest_ckpt_path)
        # save best
        if val_acc_P > best_val_acc:
            torch.save(to_save, best_ckpt_path)
        # log stats
        log_stats(stats={"train_loss (P)": train_loss_P,
                         "train_acc (P)": train_acc_P,
                         "val_loss (P)": val_loss_P,
                         "val_acc (P)": val_acc_P,
                         "alpha_a (P)": model_P.alpha_a.item(),
                         "alpha_p (P)": model_P.alpha_p.item(),
                         "lr (P)": lr_scheduler_P.get_last_lr()[0],
                         "alpha_lr (P)": lr_scheduler_P.get_last_lr()[2]},
                  log_writer=log_writer,
                  epoch=epoch,
                  args=args,
                  commit=False)
        # log_stats(stats={"train_loss (A)": train_loss_A,
        #                  "train_acc (A)": train_acc_A,
        #                  "val_loss (A)": val_loss_A,
        #                  "val_acc (A)": val_acc_A,
        #                  "alpha_a (A)": model_A.alpha_a.item(),
        #                  "alpha_p (A)": model_A.alpha_p.item(),
        #                  "lr (A)": lr_scheduler_A.get_last_lr()[0],
        #                  "alpha_lr (A)": lr_scheduler_A.get_last_lr()[2]},
        #           log_writer=log_writer,
        #           epoch=epoch,
        #           args=args,
        #           commit=False)
        log_stats(stats={"epoch": epoch},
                  log_writer=log_writer,
                  epoch=epoch,
                  args=args,
                  commit=True)

        lr_scheduler_P.step()
        # lr_scheduler_A.step()

        epoch += 1
    
    if args.use_wandb:
        wandb.finish()

if __name__ == "__main__":
    args = get_args_parser()
    args = args.parse_args()

    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    
    if args.wandb_cont_sweep:
        sweep_configuration = {
            "method": "random",
            "metric": {"goal": "maximize", "name": "val_acc (P)"},
            "parameters":
            {
                "lr": {"max": 1e-4, "min": 1e-5},
                "alpha_lr": {"max": 1e-4, "min": 1e-5},
                "cls_lr": {"max": 1e-3, "min": 1e-5},
                "aux_lr": {"max": 5e-3, "min": 1e-5},
                "r_a": {"max": 5.0, "min": 1.0},
                "r_p": {"max": 5.0, "min": 1.0},
                "lambda_1": {"values": [1.5]},
                "lambda_2": {"values": [1.5]},
                "lambda_3": {"values": [8.0]},
                "kappa_a": {"max": 5.0, "min": 1.0},
                "kappa_p": {"max": 5.0, "min": 1.0},
                "weight_decay": {"max": 3e-4, "min": 1e-5},
                "exp_lr_gamma": {"values": [0.98]}
            }
        }
        sweep_id = wandb.sweep(sweep=sweep_configuration, project=args.project_name)
        wandb.agent(sweep_id, function=main, count=10)
    else:
        main()