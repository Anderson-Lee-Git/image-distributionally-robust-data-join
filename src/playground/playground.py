import numpy as np
import os
import sys

from PIL import Image

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

md = pd.read_csv(config("TINYIMAGENET_TRAIN_META_PATH"))
md = md.loc[md["group"] == 1]  # only load group 1 for playground
path_root = config("TINYIMAGENET_TRAIN_LATENT_PATH")
embed_dim = 64
patch_size = 4
num_patches = (64 // patch_size) ** 2
d = num_patches * embed_dim

# load all latents by metadata
if not os.path.exists(os.path.join(config("REPO_ROOT"), "src/playground/sample_db.npy")):
    xb = np.ndarray(shape=(len(md), d))
    for idx in range(len(md)):
        row = md.iloc[idx]
        path = os.path.join(path_root, row["class"])
        path = os.path.join(path, row["id"][:row["id"].find(".JPEG")] + ".npy")
        xb[idx] = np.load(path).reshape(-1,)
        print_progress_bar(idx + 1, len(md), prefix='Progress:', suffix='Complete', length=50)
    np.save(os.path.join(config("REPO_ROOT"), "src/playground/sample_db.npy"), xb)
else:
    print(f"Load from npy")
    xb = np.load(os.path.join(config("REPO_ROOT"), "src/playground/sample_db.npy"))

res = faiss.StandardGpuResources()
index = faiss.IndexFlatL2(d)   # build the index
gpu_index_flat = faiss.index_cpu_to_gpu(res, 0, index)
print("Build index")
gpu_index_flat.add(xb)                  # add vectors to the index
print(gpu_index_flat.ntotal)

idx = np.random.randint(low=0, high=len(md))
q = xb[idx].reshape(1, -1)
k = 10                          # we want to see 4 nearest neighbors
D, I = gpu_index_flat.search(q, k) # sanity check
print(I)
for i in I[0]:
    row = md.iloc[i]
    print(row["class"])
    path = os.path.join(config("TINYIMAGENET_TRAIN_PATH"), row["class"])
    path = os.path.join(path, row["id"])
    img = Image.open(path)
    img.save(f"./image_{i}.png")
print(D)
