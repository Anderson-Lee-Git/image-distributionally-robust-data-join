import numpy as np
import pandas as pd
from decouple import Config, RepositoryEnv
from tqdm import tqdm

def generate_unique_random_pairs(num_pairs, len_1, len_2):
    unique_pairs = set()
    while len(unique_pairs) < num_pairs:
        i = np.random.randint(low=0, high=len_1)
        j = np.random.randint(low=0, high=len_2)
        if (i, j) not in unique_pairs:
            unique_pairs.add((i, j))
    return list(unique_pairs)

def sample_pairs(group_1: pd.DataFrame, group_2: pd.DataFrame,
                 cls_name: str, num_pairs: int, columns: list):
    sub_frame = pd.DataFrame(columns=columns)
    sub_md_1 = group_1.loc[group_1["class"] == cls_name]
    sub_md_2 = group_2.loc[group_2["class"] == cls_name]
    # draw randomly sampled indices from each metadata to forms pairs
    sample_indices = generate_unique_random_pairs(num_pairs, len(sub_md_1), len(sub_md_2))
    for i, j in sample_indices:
        id_1 = sub_md_1.iloc[i]["id"]
        id_2 = sub_md_2.iloc[j]["id"]
        class_1 = sub_md_1.iloc[i]["class"]
        class_2 = sub_md_2.iloc[j]["class"]
        label_1 = sub_md_1.iloc[i]["label"]
        label_2 = sub_md_2.iloc[j]["label"]
        sub_frame.loc[len(sub_frame)] = {
            "id_1": id_1,
            "id_2": id_2,
            "class_1": class_1,
            "class_2": class_2,
            "label_1": label_1,
            "label_2": label_2
        }
    return sub_frame

def main():
    config = Config(RepositoryEnv(".env"))
    md = pd.read_csv(config("TINYIMAGENET_TRAIN_META_PATH"))
    group_1 = md.loc[md["group"] == 1]
    group_2 = md.loc[md["group"] == 2]
    total_classes = md["class"].unique()
    pairs_per_class = 1000
    pairs_md = pd.DataFrame(columns=["id_1", "id_2", "class_1", "class_2", "label_1", "label_2"])
    # for each class
    for i in tqdm(range(len(total_classes))):
        cls_name = total_classes[i]
        sub_frame = sample_pairs(group_1, group_2, cls_name, pairs_per_class, pairs_md.columns.to_list())
        pairs_md = pd.concat([pairs_md, sub_frame])
    print(f"total number of pairs = {len(pairs_md)}")
    pairs_md.to_csv(config("TINYIMAGENET_PAIRS_META_PATH"))

if __name__ == "__main__":
    main()