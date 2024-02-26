import argparse
from pathlib import Path
import os
import sys

import wandb
import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from decouple import Config, RepositoryEnv
config = Config(RepositoryEnv(".env"))
sys.path.insert(0, config("REPO_ROOT"))

from models import DRDJVanilla
from engine import evaluate
from utils.logs import log_stats
from dataset.datasets import build_dataset, GroupCollateFnClass

def get_args_parser():
    parser = argparse.ArgumentParser('DRDJ group evaluation', add_help=False)
    parser.add_argument('--batch_size', default=64, type=int,
                        help='Batch size per GPU (effective batch size is batch_size * accum_iter * # gpus')
    # Model parameters
    parser.add_argument('--model', default='resnet50', type=str)
    parser.add_argument('--pretrained_path', default=None, type=str)
    parser.add_argument('--embed_dim', default=2048, type=int)
    parser.add_argument('--aux_embed_dim', default=32, type=int)

    # Dataset parameters
    parser.add_argument('--input_size', type=int, default=64)
    parser.add_argument('--output_dir', default='/gscratch/jamiemmt/andersonlee/image-distributionally-robust-data-join/src/misc',
                        help='path where to save, empty for no saving')
    parser.add_argument('--log_dir', default='/gscratch/jamiemmt/andersonlee/image-distributionally-robust-data-join/src/misc',
                        help='path where to tensorboard log')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--dataset', default='tiny_imagenet', type=str, help='dataset option')
    parser.add_argument('--num_classes', default=200, type=int)
    parser.add_argument('--num_groups', default=20, type=int)
    parser.add_argument('--unbalanced', action='store_true', default=False)

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
        if not os.path.exists(os.path.join(args.output_dir, "examples")):
            os.makedirs(os.path.join(args.output_dir, "examples"), exist_ok=True)
        print(f"wandb run name: {wandb.run.name}")
    device = torch.device(args.device)

    # fix seed for reproducibility
    seed = args.seed
    torch.manual_seed(seed)
    np.random.seed(seed)

    # set up dataset and data loaders
    print("Set up dataset and dataloader")
    dataset_validation = build_dataset(split='val', args=args)
    dataset_test = build_dataset(split='test', args=args)
    print(dataset_test)
    # set up log writer
    if args.log_dir is not None:
        os.makedirs(args.log_dir, exist_ok=True)
        log_writer = SummaryWriter(log_dir=args.log_dir)
    # initialize model
    model = DRDJVanilla(r_a=0, r_p=0,
                        kappa_a=0, kappa_p=0,
                        n_a=0, n_p=0,
                        lambda_1=0,
                        lambda_2=0,
                        lambda_3=0,
                        backbone=args.model,
                        num_classes=args.num_classes,
                        embed_dim=args.embed_dim,
                        aux_embed_dim=args.aux_embed_dim,
                        objective="P",
                        args=args)
    # load model
    print(f"Load drdj model {args.model} from {args.evaluate_ckpt}")
    print(f"Total evaluate images: {len(dataset_test)}")
    # load model from ckpt
    state_dict = torch.load(args.evaluate_ckpt, map_location=device)
    # print(f"Evaluate model with objective = {state_dict['objective']}")
    model.load_state_dict(state_dict[f"model_P"])

    # dataframe to store
    df = pd.DataFrame(columns=["group", "accuracy", "loss"])
    # groups
    groups = range(args.num_groups)
    # evaluate the model
    model.to(device)
    criterion = torch.nn.CrossEntropyLoss()
    """
    Validation for sanity check
    """
    data_loader_val = DataLoader(dataset_validation,
                                 batch_size=args.batch_size,
                                 num_workers=args.num_workers,
                                 collate_fn=dataset_validation.collate_fn,
                                 pin_memory=True)
    val_loss, val_acc = evaluate(model=model,
                                data_loader=data_loader_val,
                                criterion=criterion,
                                device=device,
                                args=args)
    log_stats(stats={f"overall val_loss": val_loss, f"overall val_acc": val_acc},
            epoch=None,
            log_writer=log_writer,
            args=args)
    # total eval
    data_loader_test = DataLoader(dataset_test,
                                 batch_size=args.batch_size,
                                 num_workers=args.num_workers,
                                 collate_fn=dataset_test.collate_fn,
                                 pin_memory=True)
    test_loss, test_acc = evaluate(model=model,
                                    data_loader=data_loader_test,
                                    criterion=criterion,
                                    device=device,
                                    args=args)
    # log stats
    log_stats(stats={f"overall test_loss": test_loss, f"overall test_acc": test_acc},
            epoch=None,
            log_writer=log_writer,
            args=args)
    # store in dataframe
    df.loc[len(df)] = {
        "group": -1,
        "accuracy": test_acc,
        "loss": test_loss
    }
    # group eval
    for group in groups:
        GroupCollateFnClass.group = group
        dataset_test.collate_fn = GroupCollateFnClass.group_filter_collate_fn
        data_loader_test = DataLoader(dataset_test,
                                 batch_size=args.batch_size,
                                 num_workers=args.num_workers,
                                 collate_fn=dataset_test.collate_fn,
                                 pin_memory=True)
        test_loss, test_acc = evaluate(model=model,
                                    data_loader=data_loader_test,
                                    criterion=criterion,
                                    device=device,
                                    args=args)
        # log stats
        log_stats(stats={f"test_loss ({group})": test_loss, f"test_acc ({group})": test_acc},
                epoch=None,
                log_writer=log_writer,
                args=args)
        # store in dataframe
        df.loc[len(df)] = {
            "group": group,
            "accuracy": test_acc,
            "loss": test_loss
        }
    df.to_csv(f"{args.log_dir}/{'unb_' if args.unbalanced else ''}{args.dataset}_{args.model}_drdj_vanilla_group_eval.csv")

if __name__ == "__main__":
    args = get_args_parser()
    args = args.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
    if args.use_wandb:
        wandb.finish()