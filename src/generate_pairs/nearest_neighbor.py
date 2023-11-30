import os
import argparse

import numpy as np
import pandas as pd
from decouple import Config, RepositoryEnv
config = Config(RepositoryEnv(".env"))

def get_args_parser():
    parser = argparse.ArgumentParser("Latent nearest neighbor pairing")
    return parser

def main(args):
    # load meta data
    md = pd.read_csv(config("TINYIMAGENET_TRAIN_PATH"))
    gp_1 = md.loc[md["group"] == 1]
    gp_2 = md.loc[md["group"] == 2]
    # TODO: construct indices for latents for group 1 and group 2 respectively
    
    # for each latent in group 1
        # search in group 2 indices
    # for each latent in group 2
        # search in group 1 indices
    
    # store pair list in another meta data file
    # attr: unique_id_1 (trim .JPEG)
    # attr: unique_id_2
    # attr: class id
    # attr: label (number)
    pass

if __name__ == "__main__":
    args = get_args_parser()
    args = args.parse_args()
    main(args)