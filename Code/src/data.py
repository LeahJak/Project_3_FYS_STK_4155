# Data loading for OrganAMNIST, and general helpers carried over from Project 2.
# Everyone loads data from here so it is identical across the group.
#
# Skeleton: function names and docstrings suggested by an LLM (Claude, November
# 2026); the function bodies are to be written by the group.

from pathlib import Path

import numpy as np

seed = 2026

# Facts about the dataset, from the official medmnist package (INFO["organamnist"]).
DATA_DIR = Path(__file__).resolve().parents[1] / "data"   # Code/data/
FILENAME = "organamnist.npz"
URL = "https://zenodo.org/records/10519652/files/organamnist.npz?download=1"
MD5 = "68e3f8846a6bd62f0c9bf841c0d9eacc"

LABELS = [
    "bladder", "femur-left", "femur-right", "heart", "kidney-left",
    "kidney-right", "liver", "lung-left", "lung-right", "pancreas", "spleen",
]
N_CLASSES = len(LABELS)
N_SAMPLES = {"train": 34561, "val": 6491, "test": 17778}


# --------------------------------------------------------------------------
# OrganAMNIST
# --------------------------------------------------------------------------

def download_organamnist(data_dir=DATA_DIR):
    """
    Make sure organamnist.npz is in data_dir, and return its path.

    Suggestions:
    - If the file is missing: download it from URL, or raise an error telling
      the user to download it manually and put it in Code/data/.
    - Optionally check the file against MD5, so a broken download is caught.
    """
    raise NotImplementedError


def load_train_val(flatten=True, scale=True, data_dir=DATA_DIR):
    """
    Training and validation data, for all tuning and model selection.

    Should return X_train, y_train, X_val, y_val, where
    - X has shape (n, 784) if flatten=True (FFNN, logistic regression, trees),
      or (n, 28, 28) if flatten=False (CNN);
    - pixel values are divided by 255 if scale=True;
    - y is a 1D array of integer labels 0-10 (stored as shape (n, 1) in the file).

    Keys in the npz file: train_images, train_labels, val_images, val_labels.
    """
    raise NotImplementedError


def load_test(flatten=True, scale=True, data_dir=DATA_DIR):
    """
    Test data. Use ONCE per method, in its final evaluation cell only.

    Should return X_test, y_test, in the same format as load_train_val.
    Keys in the npz file: test_images, test_labels.
    """
    raise NotImplementedError


def class_counts(y):
    """
    Number of samples in each of the 11 classes, e.g. as a dict {label name: count}.
    Hint: np.bincount with minlength=N_CLASSES.
    """
    raise NotImplementedError


def subset(X, y, n_per_class, rng):
    """
    Random subset with (at most) n_per_class samples of each class, for fast
    development and hyperparameter searches. Returns X_sub, y_sub, shuffled.
    """
    raise NotImplementedError


# --------------------------------------------------------------------------
# General helpers, carried over unchanged from Project 2
# --------------------------------------------------------------------------

def make_batches(n, batch_size, rng):
    """Shuffle the indices and split them into minibatches."""
    idx = rng.permutation(n)
    return [idx[i:i + batch_size] for i in range(0, n, batch_size)]


def scale_data(X_train, X_test):
    """
    Standardize features using the training data.
    Parameters:
    X_train : array-like. Training features.
    X_test : array-like. Test features.
    """
    mean = np.mean(X_train, axis=0)
    std = np.std(X_train, axis=0)
    std = np.where(std == 0, 1.0, std)

    X_train_scaled = (X_train - mean) / std
    X_test_scaled = (X_test - mean) / std

    return X_train_scaled, X_test_scaled, mean, std


def one_hot(y, n_classes=None):
    """Integer labels -> one-hot rows (f.ex. 3 -> [0,0,0,1,0,0,0,0,0,0])."""
    n_classes = int(np.max(y)) + 1 if n_classes is None else n_classes
    Y = np.zeros((len(y), n_classes))
    Y[np.arange(len(y)), y] = 1.0
    return Y


def stratified_kfold_indices(y, k, rng):
    """As kfold_indices, but each class is cut into k parts separately, so every fold has the class proportions of y."""
    y = np.asarray(y)
    val_parts = [[] for _ in range(k)]
    for c in np.unique(y):
        idx = rng.permutation(np.where(y == c)[0])
        for i, part in enumerate(np.array_split(idx, k)):
            val_parts[i].append(part)
    folds = []
    for i in range(k):
        val = np.concatenate(val_parts[i])
        train = np.setdiff1d(np.arange(len(y)), val)
        folds.append((train, val))
    return folds
