import numpy as np
import os
import sys
import time
import json

from PIL import Image
from tqdm import tqdm

def print_progress_bar(iteration, total, prefix='', suffix='', length=30, fill='█'):
    percent = ("{0:.1f}").format(100 * (iteration / float(total)))
    filled_length = int(length * iteration // total)
    bar = fill * filled_length + '-' * (length - filled_length)
    sys.stdout.write('\r{} |{}| {}% {}'.format(prefix, bar, percent, suffix))
    sys.stdout.flush()

import faiss                   # make faiss available
import pandas as pd
from decouple import Config, RepositoryEnv
config = Config(RepositoryEnv(".env"))

md = pd.read_csv(config("CIFAR100_TRAIN_META_PATH"))
path_root = config("CIFAR100_TRAIN_LATENT_PATH")
d = 2048

db_file = "sample_db_2048.npy"

# load all latents by metadata
if not os.path.exists(os.path.join(config("REPO_ROOT"), f"playground/{db_file}")):
    xb = np.ndarray(shape=(len(md), d))
    for idx in tqdm(range(len(md))):
        row = md.iloc[idx]
        path = os.path.join(path_root, str(row["label"]))
        path = os.path.join(path, row["id"][:row["id"].find(".jpg")] + ".npy")
        xb[idx] = np.load(path).reshape(-1,)
    np.save(os.path.join(config("REPO_ROOT"), f"playground/{db_file}"), xb)
else:
    print(f"Load from npy")
    xb = np.load(os.path.join(config("REPO_ROOT"), f"playground/{db_file}"))

print(xb[0][:5])
res = faiss.StandardGpuResources()
index = faiss.IndexFlatL2(d)   # build the index
gpu_index_flat = faiss.index_cpu_to_gpu(res, 0, index)
print("Start build index")
start = time.time()
gpu_index_flat.add(xb)                  # add vectors to the index
end = time.time()
print(f"build takes {end - start} seconds")
print(gpu_index_flat.ntotal)

superclass_map = open("/gscratch/cse/lee0618/cifar-100/meta/sub_to_superclass.json", "r")
superclass_map = json.load(superclass_map)

for step in range(3):
    idx = np.random.randint(low=0, high=len(md))
    while md.iloc[idx]["group"] != 1:
        idx = np.random.randint(low=0, high=len(md))
    q = xb[idx].reshape(1, -1)
    k = 4                          # we want to see 4 nearest neighbors
    start = time.time()
    D, I = gpu_index_flat.search(q, k) # sanity check
    end = time.time()
    print(f"search takes {end - start} seconds")
    print(f"query label = {md.iloc[idx]['label']}")
    for i in I[0]:
        row = md.iloc[i]
        print(f"value labels = {row['label']}, superclass = {superclass_map[str(row['label'])]}")
        path = os.path.join(config("CIFAR100_TRAIN_PATH"), str(row["label"]))
        path = os.path.join(path, str(row["id"]))
        img = Image.open(path)
        img.save(f"./image_{i}_{step}_{str(row['label'])}.png")
    print(I[0])
    print(D)
