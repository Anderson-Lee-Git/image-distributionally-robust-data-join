import pandas as pd
from tqdm import tqdm
import json
from decouple import Config, RepositoryEnv
config = Config(RepositoryEnv(".env"))

path = config("CIFAR100_PAIRS_UNBALANCED_META_PATH")
df = pd.read_csv(path)

md_path = config("CIFAR100_TRAIN_UNBALANCED_META_PATH")
md = pd.read_csv(md_path)

id_map = json.load(open("/gscratch/cse/lee0618/cifar-100/meta/subclass_id_to_superclass_id.json", "r"))

print(len(df))
superclass_agreement = df.loc[df["superclass_1"] == df["superclass_2"]]
label_agreement = df.loc[df["label_1"] == df["label_2"]]
for i in tqdm(range(len(df))):
    assert id_map[str(df.iloc[i]["label_1"])] == str(df.iloc[i]["superclass_1"])
    assert id_map[str(df.iloc[i]["label_2"])] == str(df.iloc[i]["superclass_2"])
#     assert md.loc[md["id"] == df.loc[i, "id_1"]].iloc[0]["group"] == 1
#     assert md.loc[md["id"] == df.loc[i, "id_2"]].iloc[0]["group"] == 2
print(len(superclass_agreement) / len(df))
print(len(label_agreement) / len(df))