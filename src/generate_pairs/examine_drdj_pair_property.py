import pandas as pd
import sklearn.preprocessing as preprocessing
import numpy as np
import sklearn
from sklearn import datasets
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KDTree

"""
Examine the pairs property in the original paper
"""

def set_k_closest_matchings(X_A, X_P, y_A, y_P, d_X, k=1):
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

    label_agreement = 0
    for idx_A, idx_P in matchings:
        if y_P[idx_P] == y_A[idx_A]:
            label_agreement += 1
    print(label_agreement / len(matchings))

    return matchings

def main():
    digits = datasets.load_digits()
    n_samples = len(digits.images)
    data = digits.images.reshape((n_samples, -1))
    print(data.shape)
    indices = []
    target_digits = [i for i in range(10)]
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
    set_k_closest_matchings(X_A, X_P, y_A, y_P, d_X, k=n_neighbors)

if __name__ == "__main__":
    main()