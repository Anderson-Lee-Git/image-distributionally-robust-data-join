import os
import argparse
import json
import pickle

import numpy as np
import pandas as pd
from tqdm import tqdm
import faiss
from decouple import Config, RepositoryEnv
config = Config(RepositoryEnv(".env"))

dataset_keys = {
    "cifar100": ("id", "label", "superclass"),
    "celebA": ("image_id", "Blond_Hair", "Male")
}

def get_args_parser():
    parser = argparse.ArgumentParser("Latent nearest neighbor pairing")
    parser.add_argument('--dataset', default=None, type=str)
    parser.add_argument('--unbalanced', action='store_true', default=False)
    parser.add_argument('--dim', default=None, type=int)
    parser.add_argument('--k', default=4, type=int, help="k nearest neighbor")
    parser.add_argument('--cached_db', action='store_true', default=True)
    return parser

def get_metadata_path(args):
    path = None
    if args.dataset == "cifar100":
        if args.unbalanced:
            path = config("CIFAR100_TRAIN_UNBALANCED_META_PATH")
        else:
            path = config("CIFAR100_TRAIN_META_PATH")
    elif args.dataset == "celebA":
        if args.unbalanced:
            path = config("CELEB_A_TRAIN_UNBALANCED_META_PATH")
        else:
            path = config("CELEB_A_TRAIN_META_PATH")
    if path is None:
        raise NotImplementedError()
    return os.path.join(config("DATASET_ROOT"), path)

def get_store_path(args):
    path = None
    if args.dataset == "cifar100":
        if args.unbalanced:
            path = config("CIFAR100_PAIRS_UNBALANCED_META_PATH")
        else:
            path = config("CIFAR100_PAIRS_META_PATH")
    elif args.dataset == "celebA":
        if args.unbalanced:
            path = config("CELEB_A_PAIRS_UNBALANCED_META_PATH")
        else:
            path = config("CELEB_A_PAIRS_META_PATH")
    if path is None:
        raise NotImplementedError()
    return os.path.join(config("DATASET_ROOT"), path)

def get_latent_path(args):
    path = None
    if args.dataset == "cifar100":
        if args.unbalanced:
            path = config("CIFAR100_TRAIN_UNBALANCED_LATENT_PATH")
        else:
            path = config("CIFAR100_TRAIN_LATENT_PATH")
    elif args.dataset == "celebA":
        if args.unbalanced:
            path = config("CELEB_A_TRAIN_UNBALANCED_LATENT_PATH")
        else:
            path = config("CELEB_A_TRAIN_LATENT_PATH")
    if path is None:
        raise NotImplementedError()
    return os.path.join(config("DATASET_ROOT"), path)

"""
Obsolete
"""
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

def get_db_from_pickle(md, db_file, pickle_path, dim, dataset, use_cache=False):
    print(f"Number of rows in metadata: {len(md)}")
    latents = pickle.load(open(pickle_path, "rb"))
    if not os.path.exists(os.path.join(config("REPO_ROOT"), f"generate_pairs/{db_file}")) or not use_cache:
        xb = np.ndarray(shape=(len(md), dim))
        for idx in tqdm(range(len(md))):
            row = md.iloc[idx]
            id_key, _, _ = dataset_keys[dataset]
            image_id = row[id_key]
            xb[idx] = latents[image_id].reshape(-1,)
        np.save(os.path.join(config("REPO_ROOT"), f"generate_pairs/{db_file}"), xb)
    else:
        print(f"[INFO] Load from npy")
        xb = np.load(os.path.join(config("REPO_ROOT"), f"generate_pairs/{db_file}"))
    return xb

def generate_pairs(query_md, target_md, query_db, target_db, dim, k, dataset):
    res_df = pd.DataFrame(columns=["id_q", "id_t", "label_q", "label_t", "aux_q", "aux_t", "dist"])
    res = faiss.StandardGpuResources()
    index = faiss.IndexFlatL2(dim)
    gpu_index_flat = faiss.index_cpu_to_gpu(res, 0, index)
    gpu_index_flat.add(target_db)
    print(f"Number of indices: {gpu_index_flat.ntotal}")
    D, I = gpu_index_flat.search(query_db, k)
    id_key, label_key, aux_key = dataset_keys[dataset]
    for idx_q in tqdm(range(len(I))):
        for idx_t in I[idx_q]:
            if idx_t != -1:
                row_q = query_md.iloc[idx_q]
                row_t = target_md.iloc[idx_t]
                res_df.loc[len(res_df)] = \
                    {"id_q": row_q[id_key],
                    "id_t": row_t[id_key],
                    "label_q": row_q[label_key],
                    "label_t": row_t[label_key],
                    "aux_q": row_q[aux_key],
                    "aux_t": row_t[aux_key],
                    "dist": D[idx_q][0]}
    return res_df

def main(args):
    # load meta data
    md = pd.read_csv(get_metadata_path(args))
    gp_1 = pd.DataFrame(md.loc[md["group"] == 1])
    gp_2 = pd.DataFrame(md.loc[md["group"] == 2])
    pickle_path = get_latent_path(args)
    dim = args.dim
    k = args.k
    db_file_1 = f"{'unb_' if args.unbalanced else ''}{args.dataset}_dim_{dim}_k_{k}_group_1.npy"
    db_file_2 = f"{'unb_' if args.unbalanced else ''}{args.dataset}_dim_{dim}_k_{k}_group_2.npy"
    xb_1 = get_db_from_pickle(md=gp_1, db_file=db_file_1, pickle_path=pickle_path, dim=dim, dataset=args.dataset)
    xb_2 = get_db_from_pickle(md=gp_2, db_file=db_file_2, pickle_path=pickle_path, dim=dim, dataset=args.dataset)
    df_1 = generate_pairs(query_md=gp_1,
                          target_md=gp_2,
                          query_db=xb_1,
                          target_db=xb_2,
                          dim=dim,
                          k=k,
                          dataset=args.dataset)
    df_2 = generate_pairs(query_md=gp_2,
                          target_md=gp_1,
                          query_db=xb_2,
                          target_db=xb_1,
                          dim=dim,
                          k=k,
                          dataset=args.dataset)

    res_df = pd.DataFrame(columns=["id_1", "id_2", "label_1", "label_2", "aux_1", "aux_2", "dist"])
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
                "aux_1": row["aux_q"],
                "aux_2": row["aux_t"],
                "dist": row["dist"]
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
                "aux_1": row["aux_t"],
                "aux_2": row["aux_q"],
                "dist": row["dist"]
            }

    # store pair list in another meta data file
    # attr: unique_id_1 (trim .JPEG)
    # attr: unique_id_2
    # attr: class id
    # attr: label (number)
    print(res_df.iloc[:10])
    print(f"total number of pairs: {len(res_df)}")
    res_df.to_csv(get_store_path(args))
    # res_df.to_csv("/gscratch/jamiemmt/andersonlee/image-distributionally-robust-data-join/src/generate_pairs/test_pairs.csv")

if __name__ == "__main__":
    args = get_args_parser()
    args = args.parse_args()
    main(args)