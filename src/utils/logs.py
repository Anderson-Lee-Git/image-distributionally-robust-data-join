import json
import os

import wandb
from torch.utils.tensorboard import SummaryWriter

def log_stats(stats: dict, log_writer: SummaryWriter, epoch: int, args, commit=True):
    if log_writer is not None:
        log_writer.flush()
    with open(os.path.join(args.output_dir, "log.txt"), mode="a", encoding="utf-8") as f:
        f.write(json.dumps(stats) + "\n")
    if log_writer is not None:
        for metric in stats:
            # train
            if epoch is not None:
                log_writer.add_scalar(metric, stats[metric], epoch)
            print(f"{metric}: {stats[metric]}", end=" ")
        print()
            
    if args.use_wandb:
        wandb.log(stats, commit=commit)

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)