import os
import pandas as pd
from tqdm import tqdm
import json
from decouple import Config, RepositoryEnv
config = Config(RepositoryEnv(".env"))

path = os.path.join(config("DATASET_ROOT"), config("CELEB_A_PAIRS_META_PATH"))
# path = config("CIFAR100_PAIRS_META_PATH")
df = pd.read_csv(path)

md_path = config("CIFAR100_TRAIN_UNBALANCED_META_PATH")
md = pd.read_csv(md_path)

print(len(df))
aux_agreement = df.loc[df["aux_1"] == df["aux_2"]]
label_agreement = df.loc[df["label_1"] == df["label_2"]]
for i in tqdm(range(len(df))):
    assert md.loc[md["id"] == df.loc[i, "id_1"]].iloc[0]["group"] == 1
    assert md.loc[md["id"] == df.loc[i, "id_2"]].iloc[0]["group"] == 2
print(len(aux_agreement) / len(df))
print(len(label_agreement) / len(df))