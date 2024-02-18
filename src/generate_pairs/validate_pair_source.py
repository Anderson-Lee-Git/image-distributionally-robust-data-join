import pandas as pd
from tqdm import tqdm

path = "/gscratch/cse/lee0618/cifar-100/meta/pairs.csv"
df = pd.read_csv(path)

md_path = "/gscratch/cse/lee0618/cifar-100/meta/train.csv"
md = pd.read_csv(md_path)

print(len(df))
superclass_agreement = df.loc[df["superclass_1"] == df["superclass_2"]]
label_agreement = df.loc[df["label_1"] == df["label_2"]]
# for i in tqdm(range(len(df))):
#     assert md.loc[md["id"] == df.loc[i, "id_1"]].iloc[0]["group"] == 1
#     assert md.loc[md["id"] == df.loc[i, "id_2"]].iloc[0]["group"] == 2
print(len(superclass_agreement) / len(df))
print(len(label_agreement) / len(df))