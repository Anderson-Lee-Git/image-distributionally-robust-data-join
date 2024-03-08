import pandas as pd
import sklearn.preprocessing as preprocessing
import numpy as np
import sklearn
from sklearn import datasets
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KDTree
import matplotlib.pyplot as plt

"""
Examine the pairs property in the original paper
"""

def set_k_closest_matchings(X_A, X_P, y_A, y_P, d_X, k=1):
    n_A, _ = X_A[:, :d_X].shape
    n_P, _ = X_P.shape

    # Handle X_A's
    k_closest_neighbors_A = KDTree(X_A[:, :d_X])
    dist_from_A, neighbors_from_A = k_closest_neighbors_A.query(X_P[:, :d_X], k, return_distance=True)

    # Handle X_P's
    k_closest_neighbors_P = KDTree(X_P)
    dist_from_P, neighbors_from_P = k_closest_neighbors_P.query(X_A[:, :d_X], k, return_distance=True)

    matchings_list = []
    for j in range(len(neighbors_from_A)):
        neighbors = neighbors_from_A[j]
        dists = dist_from_A[j]
        [matchings_list.append((neighbor, j, d)) for neighbor, d in zip(neighbors, dists)]

    for i in range(len(neighbors_from_P)):
        neighbors = neighbors_from_P[i]
        dists = dist_from_P[i]
        [matchings_list.append((i, neighbor, d)) for neighbor, d in zip(neighbors, dists)]

    pair_idx_set = set()
    unique_matchings = []
    for i, j, d in matchings_list:
        if (i, j) not in pair_idx_set:
            unique_matchings.append((i, j, d))
            pair_idx_set.add((i, j))

    label_agreement = 0
    for idx_A, idx_P, dist in unique_matchings:
        if y_P[idx_P] == y_A[idx_A]:
            label_agreement += 1
    print(label_agreement / len(unique_matchings))

    return unique_matchings

def main():
    digits = datasets.load_digits()
    n_samples = len(digits.images)
    data = digits.images.reshape((n_samples, -1))
    print(data.shape)
    indices = []
    target_digits = [1, 8]
    for target in target_digits:
        indices.append(digits.target == target)
    X = np.concatenate([data[indice] for indice in indices])
    y = np.concatenate([digits.target[indice] for indice in indices])
    scaler = preprocessing.StandardScaler()
    X = scaler.fit_transform(X)
    # PARAMETERS
    d_X = 32
    n_P = 30
    overlapped = 0
    n_neighbors = 1
    seed = 42
    X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, stratify=y, test_size=0.3, random_state=seed)
    X_A = X_train[n_P - overlapped:, :]
    y_A = y_train[n_P - overlapped:]
    X_P = X_train[:n_P, :d_X]
    y_P = y_train[:n_P]
    matchings = set_k_closest_matchings(X_A, X_P, y_A, y_P, d_X, k=n_neighbors)
    agreement = np.array([1 if y_A[i] == y_P[j] else 0 for (i, j, _) in matchings])
    disagreement = np.array([1 if y_A[i] != y_P[j] else 0 for (i, j, _) in matchings])
    dist = np.array([d for (i, j, d) in matchings])
    agree_dist = dist.compress(agreement, axis=0)
    disagree_dist = dist.compress(disagreement, axis=0)
    cor = np.corrcoef(x=np.stack([dist, agreement]))
    print(f"Correlation = {cor}")
    x = np.array([0, 1])

    fig = plt.figure(figsize=(5, 8))
    y_means = np.array([np.mean(disagree_dist), np.mean(agree_dist)])
    y_stds = np.array([np.std(disagree_dist), np.std(agree_dist)])
    plt.scatter(x=np.zeros(len(disagree_dist)), y=disagree_dist, alpha=0.5, facecolor='none', s=30, edgecolor='tab:orange')
    plt.scatter(x=np.ones(len(agree_dist)), y=agree_dist, alpha=0.5, facecolor='none', s=30, edgecolor='tab:blue')
    plt.errorbar(x, y_means, yerr=y_stds, fmt='s', capsize=10, color="black")
    plt.xticks([0, 1], ['Disagree', 'Agree'])
    plt.ylabel('Distance')
    plt.xlim(-0.5, 1.5)
    plt.grid(True)
    plt.savefig("/gscratch/jamiemmt/andersonlee/image-distributionally-robust-data-join/src/generate_pairs/cor.png")
    
    
if __name__ == "__main__":
    main()