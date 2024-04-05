import argparse
import yaml
import os
import json
from decouple import Config, RepositoryEnv

from gpuscheduler import HyakScheduler

parser = argparse.ArgumentParser(description='Compute script.')
parser.add_argument('--dry', action='store_true')
parser.add_argument('--verbose', action='store_true')
parser.add_argument('--p', type=float, default=1.0, help='Probability with which to select a configuration.')
parser.add_argument('-A', type=str, default="stf")
parser.add_argument('-p', type=str, default="ckpt")
args = parser.parse_args()

config = Config(RepositoryEnv(".env"))

model = "resnet50"
dataset = "cifar100_pairs"

account = args.A
partition = args.p

# sbatch details
gpus = 1
cmd = "wandb agent --count 1 "
name = f"drdj_{model}_{dataset}_{partition}"
cores_per_job = 5
mem = 64
time_hours = 24
time_minutes = 0
constraint = ""
exclude = ""

repo = config("GIT_HOME")
change_dir = config("GIT_HOME")
scheduler = HyakScheduler(verbose=args.verbose, use_wandb=True, exp_name=name, account=account, partition=partition)
ckpt_base_dir = config("LOG_HOME")
logfolder = os.path.join(ckpt_base_dir, name)
sweep_config_path = config("SWEEP_CONFIG_BASE_PATH")
num_runs = 10

# default commands and args
epochs = 8
batch_size = 48
input_size = 224
num_workers = 5
data_subset = 1.0
data_group = 1
num_classes = 100
freeze_param = False
aux_embed_dim = 20
base_flags = [
    "${env}",
    "python",
    "drdj_vanilla/train.py",
    "--use_wandb",
    f"--project_name={name}",
    f"--output_dir={logfolder}",
    f"--log_dir={logfolder}",
    f"--model={model}",
    f"--batch_size={batch_size}",
    f"--epochs={epochs}",
    f"--input_size={input_size}",
    f"--num_workers={num_workers}",
    f"--data_subset={data_subset}",
    f"--dataset={dataset}",
    f"--data_group={data_group}",
    f"--num_classes={num_classes}",
    f"--aux_embed_dim={aux_embed_dim}",
    # "--unbalanced",
    "${args}"  # use args from configuration as command arguments
]

sweep_configuration = {
    "method": "random",
    "metric": {"goal": "maximize", "name": "val_acc (P)"},
    "parameters":
    {
        "lr": {"max": 1e-3, "min": 1e-5},
        "alpha_lr": {"max": 1e-4, "min": 1e-5},
        "cls_lr": {"max": 1e-3, "min": 1e-5},
        "r_a": {"values": [0.0]},
        "r_p": {"values": [0.0]},
        "lambda_1": {"values": [1.0]},
        "lambda_2": {"values": [1.0]},
        "lambda_3": {"values": [1.0]},
        "kappa_a": {"max": 5.0, "min": 1.0},
        "kappa_p": {"max": 5.0, "min": 1.0},
        "weight_decay": {"max": 3e-4, "min": 1e-5},
        "exp_lr_gamma": {"values": [0.99]}
    },
    "command": base_flags
}

# Create log folder
if not os.path.exists(logfolder):
    print(f"Creating {logfolder}")
    os.makedirs(logfolder)

# remove previous sweep output if one exists
sweep_out_file = f'{logfolder}/sweepid.txt'
if os.path.exists(sweep_out_file):
    os.remove(sweep_out_file)

# dump sweep config for main to read
with open(f"{sweep_config_path}/{name}.yaml", "w") as config_file:
    yaml.dump(sweep_configuration, config_file)

# add job to scheduler
for i in range(num_runs):
    scheduler.add_job(logfolder, change_dir, [cmd], time_hours, cores=cores_per_job, mem=mem, constraint=constraint, exclude=exclude, time_minutes=time_minutes, gpus=gpus)

if args.dry:
    print(sweep_configuration)
else:
    scheduler.run_jobs(begin=None, single_process=True)
