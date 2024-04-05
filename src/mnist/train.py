import os
import sys
import random

from tqdm import tqdm
from sklearn import datasets
from torchvision import transforms
import torch
from torch import nn as nn
from torch.optim import AdamW, SGD
from sklearn.neighbors import KDTree
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np
from decouple import Config, RepositoryEnv
config = Config(RepositoryEnv(".env"))
sys.path.insert(0, config("REPO_ROOT"))

from MnistDRDJ import MnistDRDJ
from DRDJ_engine import train_one_epoch as train_DRDJ
from DRDJ_engine import evaluate as eval_DRDJ
from baseline_engine import train_one_epoch as train_baseline
from baseline_engine import evaluate as eval_baseline

def set_seed(seed=42):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

def set_k_closest_matchings(X_A, X_P, d_X, k=5):
    n_A, _ = X_A[:, :d_X].shape
    n_P, _ = X_P.shape

    # Handle X_A's
    k_closest_neighbors_A = KDTree(X_A[:, :d_X])
    neighbors_from_A = k_closest_neighbors_A.query(X_P[:, :d_X], k, return_distance=False)

    # Handle X_P's
    k_closest_neighbors_P = KDTree(X_P)
    neighbors_from_P = k_closest_neighbors_P.query(X_A[:, :d_X], k, return_distance=False)

    matchings_list = []
    for (j, neighbors) in enumerate(neighbors_from_A):
        [matchings_list.append((neighbor, j)) for neighbor in neighbors]

    for (i, neighbors) in enumerate(neighbors_from_P):
        [matchings_list.append((i, neighbor)) for neighbor in neighbors]

    matchings = np.array(list(set(matchings_list)))
    return matchings

def generate_pairs(X, y):
    one_mask = y == 1
    eight_mask = y == 8
    X = X[one_mask | eight_mask]
    y = y[one_mask | eight_mask]
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2)
    n_P = 60
    n_A = len(X_train) - n_P
    d_X = 32
    X_A = X_train[n_P:]
    y_A = y_train[n_P:]
    X_P = X_train[:n_P, :d_X]
    y_P = y_train[:n_P]
    # make to 0 and 1 labels
    y_P = (y_P == 8).astype(int)
    y_test = (y_test == 8).astype(int)
    matchings = set_k_closest_matchings(X_A, X_P, d_X, k=1)
    print(X_A.shape)
    print(X_P.shape)
    return {
        "X_A": X_A,
        "X_P": X_P,
        "X_test": X_test,
        "y_A": y_A,
        "y_P": y_P,
        "y_test": y_test,
        "n_A": n_A,
        "n_P": n_P,
        "matchings": matchings,
        "d_X": 32
    }

def get_model(n_A, n_P, objective):
    return MnistDRDJ(r_a=1.85,
                    r_p=1.85,
                    kappa_a=5.0,
                    kappa_p=5.0,
                    n_a=n_A,
                    n_p=n_P,
                    lambda_1=1.0,
                    lambda_2=3.0,
                    lambda_3=10.0,
                    num_classes=2,
                    embed_dim=16,
                    aux_embed_dim=16,
                    objective=objective,
                    args={})

def get_baseline(num_classes):
    return nn.Sequential(
        nn.Linear(32, 64),
        # nn.LeakyReLU(),
        # nn.Linear(64, 128),
        # nn.LeakyReLU(),
        # nn.Linear(128, 64),
        nn.LeakyReLU(),
        nn.Linear(64, 16),
        nn.Linear(16, num_classes)
    )


def main():
    set_seed()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    digits = datasets.load_digits()
    n = len(digits.data)
    X = digits.data
    y = digits.target
    res = generate_pairs(X, y)
    X_P = res["X_P"]
    y_P = res["y_P"]
    X_A = res["X_A"]
    n_A = res["n_A"]
    n_P = res["n_P"]
    d_X = res["d_X"]
    X_test = res["X_test"]
    y_test = res["y_test"]
    matchings = res["matchings"]

    batch_size = 16
    epochs = 20

    # baseline
    lrs = torch.linspace(1e-3, 1e-1, steps=20)
    best_acc_baseline = 0
    for lr in lrs:
        model = get_baseline(2).to(device)
        optimizer = SGD(model.parameters(), lr=lr)
        for i in range(1, epochs+1):
            # print(f"Epoch: {i}")
            acc = train_baseline(X_P=X_P, y_P=y_P, model=model,
                                optimizer=optimizer, batch_size=batch_size,
                                device=device)
            # print(f"train accuracy = {(acc * 100):.2f}")
            test_acc = eval_baseline(X_test[:, :d_X], y_test, model, batch_size, device)
            # print(f"test accuracy = {(test_acc * 100):.2f}")
            best_acc_baseline = max(best_acc_baseline, test_acc)

    # DRDJ
    lrs = torch.linspace(1e-3, 1e-1, steps=20)
    best_acc_DRDJ = 0
    for lr in lrs:
        model_P = get_model(n_A, n_P, "P").to(device)
        optimizer_P = SGD(model_P.parameters(), lr=lr)
        model_A = get_model(n_A, n_P, "A").to(device)
        optimizer_A = SGD(model_A.parameters(), lr=lr)
        for i in range(1, epochs+1):
            # print(f"Epoch: {i}")
            acc_P, acc_A = train_DRDJ(X_P=X_P, X_A=X_A, y_P=y_P, d_X=d_X,
                                    matchings=matchings, model_P=model_P, model_A=model_A,
                                    optimizer_P=optimizer_P, optimizer_A=optimizer_A,
                                    batch_size=batch_size, device=device)
            # print(f"train accuracy (P) = {(acc_P * 100):.2f}")
            # print(f"train accuracy (A) = {(acc_A * 100):.2f}")
            if acc_A < acc_P:
                model = model_A
            else:
                model = model_P
            test_acc = eval_DRDJ(X_test=X_test, y_test=y_test, d_X=d_X, 
                                model=model, batch_size=batch_size, device=device)
            # print(f"test accuracy = {(test_acc * 100):.2f}")
            best_acc_DRDJ = max(best_acc_DRDJ, test_acc)
        print(f"penalty = {model._penalty().item()}")
        
    print(f"best baseline: {(best_acc_baseline * 100):.2f}")
    print(f"best DRDJ: {(best_acc_DRDJ * 100):.2f}")
    
if __name__ == "__main__":
    main()
