from argparse import Namespace

def get_wandb_args(run, args):
    args = vars(args)
    config = run.config
    for key in args:
        if key in config:
            args[key] = config[key]
    return Namespace(**args)

def load_saved_args(saved_args, current_args):
    saved_args = vars(saved_args)
    current_args = vars(current_args)
    except_keys = ["project_name", "seed", "output_dir", "log_dir", "epochs"]
    for key in current_args:
        if key in saved_args and key not in except_keys:
            current_args[key] = saved_args[key]
    return Namespace(**current_args)