import argparse
import json
import os
import sys

from decouple import Config, RepositoryEnv
config = Config(RepositoryEnv(".env"))
sys.path.insert(0, config("REPO_ROOT"))

import torch
from torch.utils.data import DataLoader
import pandas as pd

from drdj_adversarial.engine import evaluate as drdj_adversarial_evaluate
from drdj_vanilla.engine import evaluate as drdj_vanilla_evaluate
from baseline.engine import evaluate as baseline_evaluate
from models import DRDJAdversarial, DRDJVanilla, build_resnet

from dataset import build_dataset

def get_args_parser():
    task_args = json.load(open(os.path.join(config("REPO_ROOT"), "misc/task.json"), "r"))
    parser = argparse.ArgumentParser('Bulk Evaluation', add_help=False)
    for key in task_args:
        parser.add_argument(key, default=task_args[key])
    return parser

def load_model(args):
    if args.type == "baseline":
        model = build_resnet(num_classes=args.num_classes, args=args)
    elif args.type == "drdj_vanilla":
        model = DRDJVanilla(r_a=0, r_p=0,
                            kappa_a=0, kappa_p=0,
                            n_a=0, n_p=0,
                            lambda_1=0,
                            lambda_2=0,
                            lambda_3=0,
                            backbone=args.model,
                            num_classes=args.num_classes,
                            args=args)
    elif args.type == "drdj_adversarial":
        model = DRDJAdversarial(r_a=0, r_p=0,
                                adv_lr=0,
                                backbone=args.model,
                                num_classes=args.num_classes,
                                args=args)
    else:
        raise NotImplementedError("Not supported")
    return model

def find_ckpt_list(sweep_dir: str):
    ckpt_list = []
    for sweep_folder in os.listdir(sweep_dir):
        path = os.path.join(sweep_dir, sweep_folder)
        if os.path.isdir(path):
            # iterate through all ckpts
            latest = 0
            for ckpt in os.listdir(path):
                if ckpt.endswith(".pth"):
                    latest = max(latest, int(ckpt[ckpt.find("-")+1:ckpt.find(".")]))
            if latest > 0:
                ckpt_list.append(
                    os.path.join(path, f"checkpoint-{latest}.pth")
                )
    return ckpt_list

def evaluate(model,
            data_loader,
            criterion,
            device,
            args):
    if args.type == "baseline":
        return baseline_evaluate(model=model,
                                data_loader=data_loader,
                                criterion=criterion,
                                device=device,
                                args=args)
    elif args.type == "drdj_vanilla":
        return drdj_vanilla_evaluate(model=model,
                                    data_loader=data_loader,
                                    criterion=criterion,
                                    device=device,
                                    args=args)
    elif args.type == "drdj_adversarial":
        return drdj_adversarial_evaluate(model=model,
                                        data_loader=data_loader,
                                        criterion=criterion,
                                        device=device,
                                        args=args)
    else:
        raise NotImplementedError("Not supported")

def main(args):
    misc_folder = "/gscratch/jamiemmt/andersonlee/image-distributionally-robust-data-join/src/misc/"
    if os.path.exists(misc_folder + f"{args.name}.csv"):
        df = pd.read_csv(misc_folder + f"{args.name}.csv")
        print(f"Found existing csv with length = {len(df)}")
    else:
        df = pd.DataFrame(
            columns=[
                "regular_test_accuracy",
                "easy_corruption_accuracy",
                "hard_corruption_accuracy",
                "easy_drop",
                "hard_drop",
                "ckpt"
            ]
        )
    ckpt_list = find_ckpt_list(args.sweep_dir)
    device = torch.device(args.device)
    model = load_model(args)
    criterion = torch.nn.CrossEntropyLoss()
    datasets = [("cifar100", 1), ("cifar100_c", 1), ("cifar100_c", 2)]
    for j, ckpt in enumerate(ckpt_list):
        print(f"Checkpoint {j+1} / {len(ckpt_list)}")
        state_dict = torch.load(ckpt, map_location=device)
        model.load_state_dict(state_dict["model"], strict=False)
        model.to(device)
        row = {}
        for i, dataset in enumerate(datasets):
            args.dataset, args.severity = dataset
            args.corruption = "all"
            dataset_test = build_dataset(split='test', args=args)
            data_loader_test = DataLoader(dataset_test,
                                        batch_size=args.batch_size,
                                        num_workers=args.num_workers,
                                        pin_memory=True)
            print(dataset_test)
            test_loss, test_acc = evaluate(model=model,
                                           data_loader=data_loader_test,
                                           criterion=criterion,
                                           device=device,
                                           args=args)
            if i == 0:
                row["regular_test_accuracy"] = test_acc
            elif i == 1:
                row["easy_corruption_accuracy"] = test_acc
            else:
                row["hard_corruption_accuracy"] = test_acc
        row["easy_drop"] = row["easy_corruption_accuracy"] - row["regular_test_accuracy"]
        row["hard_drop"] = row["hard_corruption_accuracy"] - row["regular_test_accuracy"]
        row["ckpt"] = ckpt
        df.loc[len(df)] = row
    df.to_csv(f"{misc_folder}/{args.name}.csv")

if __name__ == "__main__":
    args = get_args_parser()
    args = args.parse_args()
    main(args)