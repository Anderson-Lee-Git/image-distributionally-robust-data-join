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

from models import DRDJAdversarial
from engine import evaluate
from utils.logs import log_stats
from dataset.datasets import build_dataset

def get_args_parser():
    parser = argparse.ArgumentParser('DRDJ optimization', add_help=False)
    parser.add_argument('--batch_size', default=64, type=int,
                        help='Batch size per GPU (effective batch size is batch_size * accum_iter * # gpus')
    # Model parameters
    parser.add_argument('--model', default='ResNet50', type=str)
    parser.add_argument('--pretrained_path', default=None, type=str)

    # Dataset parameters
    parser.add_argument('--input_size', type=int, default=64)
    parser.add_argument('--output_dir', default='./output_dir',
                        help='path where to save, empty for no saving')
    parser.add_argument('--log_dir', default='./output_dir',
                        help='path where to tensorboard log')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--dataset', default='tiny_imagenet', type=str, help='dataset option')
    parser.add_argument('--num_classes', default=200, type=int)

    parser.add_argument('--num_workers', default=5, type=int)
    parser.add_argument('--data_group', default=1, type=int)
    parser.add_argument('--corruption', default=None, type=str)
    parser.add_argument('--severity', default=1, type=int)

    # misc
    parser.add_argument('--use_wandb', action='store_true', default=False)
    parser.add_argument('--project_name', default='', type=str,
                        help='wandb project name')
    
    # evaluation
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
    dataset_test = build_dataset(split='test', args=args)
    print(dataset_test)
    data_loader_test = DataLoader(dataset_test,
                                 batch_size=args.batch_size,
                                 num_workers=args.num_workers,
                                 pin_memory=True)
    # set up log writer
    if args.log_dir is not None:
        os.makedirs(args.log_dir, exist_ok=True)
        log_writer = SummaryWriter(log_dir=args.log_dir)
    # initialize model
    model = DRDJAdversarial(r_a=0, r_p=0,
                            adv_lr=0,
                            backbone=args.model,
                            num_classes=args.num_classes,
                            args=args)
    # load model
    print(f"Load drdj model {args.model} from {args.evaluate_ckpt}")
    print(f"Total evaluate images: {len(dataset_test)}")
    # load model from ckpt
    state_dict = torch.load(args.evaluate_ckpt, map_location=device)
    model.load_state_dict(state_dict["model"], strict=False)
    # evaluate the model
    model.to(device)
    criterion = torch.nn.CrossEntropyLoss()
    test_loss, test_acc = evaluate(model=model,
                                   data_loader=data_loader_test,
                                   criterion=criterion,
                                   device=device,
                                   args=args)
    # log stats
    log_stats(stats={"test_loss": test_loss, "test_acc": test_acc},
             epoch=None,
             log_writer=log_writer,
             args=args)

if __name__ == "__main__":
    args = get_args_parser()
    args = args.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
    if args.use_wandb:
        wandb.finish()