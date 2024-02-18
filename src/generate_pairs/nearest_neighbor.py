import os
import argparse
import json

import numpy as np
import pandas as pd
from tqdm import tqdm
import faiss
from decouple import Config, RepositoryEnv
config = Config(RepositoryEnv(".env"))

def get_args_parser():
    parser = argparse.ArgumentParser("Latent nearest neighbor pairing")
    parser.add_argument('--dataset', default=None, type=str)
    parser.add_argument('--dim', default=None, type=int)
    parser.add_argument('--k', default=4, type=int, help="k nearest neighbor")
    return parser

def get_metadata_path(args):
    if args.dataset == "cifar100":
        return config("CIFAR100_TRAIN_META_PATH")

def get_latent_path(args):
    if args.dataset == "cifar100":
        return config("CIFAR100_TRAIN_LATENT_PATH")

def get_db(md, db_file, path_root, dim):
    print(f"Number of rows in metadata: {len(md)}")
    if not os.path.exists(os.path.join(config("REPO_ROOT"), f"generate_pairs/{db_file}")):
        xb = np.ndarray(shape=(len(md), dim))
        for idx in tqdm(range(len(md))):
            row = md.iloc[idx]
            path = os.path.join(path_root, str(row["label"]))
            path = os.path.join(path, str(row["id"][:row["id"].find(".")]) + ".npy")
            xb[idx] = np.load(path).reshape(-1,)
        np.save(os.path.join(config("REPO_ROOT"), f"generate_pairs/{db_file}"), xb)
    else:
        print(f"[INFO] Load from npy")
        xb = np.load(os.path.join(config("REPO_ROOT"), f"generate_pairs/{db_file}"))
    return xb

def generate_pairs(query_md, target_md, query_db, target_db, dim, k):
    res_df = pd.DataFrame(columns=["id_q", "id_t", "label_q", "label_t"])
    res = faiss.StandardGpuResources()
    index = faiss.IndexFlatL2(dim)
    gpu_index_flat = faiss.index_cpu_to_gpu(res, 0, index)
    gpu_index_flat.add(target_db)
    print(f"Number of indices: {gpu_index_flat.ntotal}")
    D, I = gpu_index_flat.search(query_db, k)
    for idx_q in tqdm(range(len(I))):
        for idx_t in I[idx_q]:
            if idx_t != -1:
                res_df.loc[len(res_df)] = \
                    {"id_q": query_md.iloc[idx_q]["id"],
                    "id_t": target_md.iloc[idx_t]["id"],
                    "label_q": query_md.iloc[idx_q]["label"],
                    "label_t": target_md.iloc[idx_t]["label"]}
    return res_df

def main(args):
    # load meta data
    md = pd.read_csv(get_metadata_path(args))
    gp_1 = md.loc[md["group"] == 1]
    gp_2 = md.loc[md["group"] == 2]
    # TODO: construct indices for latents for group 1 and group 2 respectively
    path_root = get_latent_path(args)
    dim = args.dim
    k = args.k
    db_file_1 = f"{args.dataset}_dim_{dim}_k_{k}_group_1.npy"
    db_file_2 = f"{args.dataset}_dim_{dim}_k_{k}_group_2.npy"
    xb_1 = get_db(md=gp_1, db_file=db_file_1, path_root=path_root, dim=dim)
    xb_2 = get_db(md=gp_2, db_file=db_file_2, path_root=path_root, dim=dim)
    df_1 = generate_pairs(query_md=gp_1,
                          target_md=gp_2,
                          query_db=xb_1,
                          target_db=xb_2,
                          dim=dim,
                          k=k)
    df_2 = generate_pairs(query_md=gp_2,
                          target_md=gp_1,
                          query_db=xb_2,
                          target_db=xb_1,
                          dim=dim,
                          k=k)
    # cifar100 only
    superclass_map = open("/gscratch/cse/lee0618/cifar-100/meta/subclass_id_to_superclass_id.json", "r")
    superclass_map = json.load(superclass_map)

    res_df = pd.DataFrame(columns=["id_1", "id_2", "label_1", "label_2", "superclass_1", "superclass_2"])
    pair_set = set()
    print(f"[INFO] Aggregating dataframes")
    for idx in tqdm(range(len(df_1))):
        row = df_1.iloc[idx]
        id_q, id_t = row["id_q"], row["id_t"]

        if (id_q, id_t) not in pair_set and (id_t, id_q) not in pair_set:
            pair_set.add((id_q, id_t))
            res_df.loc[len(res_df)] = {
                "id_1": id_q,
                "id_2": id_t,
                "label_1": row["label_q"],
                "label_2": row["label_t"],
                "superclass_1": superclass_map[str(row["label_q"])],
                "superclass_2": superclass_map[str(row["label_t"])]
            }
    for idx in tqdm(range(len(df_2))):
        row = df_2.iloc[idx]
        id_q, id_t = row["id_q"], row["id_t"]
        if (id_q, id_t) not in pair_set and (id_t, id_q) not in pair_set:
            pair_set.add((id_q, id_t))
            res_df.loc[len(res_df)] = {
                "id_1": id_t,
                "id_2": id_q,
                "label_1": row["label_t"],
                "label_2": row["label_q"],
                "superclass_1": superclass_map[str(row["label_t"])],
                "superclass_2": superclass_map[str(row["label_q"])]
            }

    # store pair list in another meta data file
    # attr: unique_id_1 (trim .JPEG)
    # attr: unique_id_2
    # attr: class id
    # attr: label (number)
    print(res_df.iloc[:10])
    res_df.to_csv(config("CIFAR100_PAIRS_META_PATH"))

if __name__ == "__main__":
    args = get_args_parser()
    args = args.parse_args()
    main(args)