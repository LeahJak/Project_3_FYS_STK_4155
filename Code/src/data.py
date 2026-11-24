# Data loading for OrganAMNIST, and general helpers carried over from Project 2.
# Everyone loads data from here so it is identical across the group.
#
# Skeleton: function names and docstrings suggested by an LLM (Claude, November
# 2026); the function bodies are to be written by the group.

import hashlib  # digital fingerprint 
import urllib.request
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

def _md5(path):
    """
    Calculate the fingerprint for the file using the MD5 hash algorithm

    LLM-assisted
        ------------
        Tool: Claude (Claude Opus 5.5, claude.ai, November 2026)
        Role: Wrote the loop with the bitwise left shift to save memory when calculating checksum. 
    """
    h = hashlib.md5()   # empty hash object 
    with open(path, "rb") as f:     #open file binary-read 
        for chunk in iter(lambda: f.read(1 << 20), b""): # Bitwise read until sentinel value is found (b)
            h.update(chunk) # update the hash 
    return h.hexdigest() # Return final hash as hexadecimal sum. 

def download_organamnist():
    """
    Download organamnist.npz (about 38 MB) from Zenodo to Code/data/ if it is
    not there already, and check its MD5 checksum.

    LLM-assisted
            ------------
            Tool: Claude (Claude Opus 5.5, claude.ai, November 2026)
            Role: Edited the code block to include the check is the dataset is present. 
                  Gave instructions on how to use urllib. 
    """
    data_dir = DATA_DIR
    path = data_dir / FILENAME                          # construct file path 
    if not path.exists():                               # Check if the file exists
        data_dir.mkdir(parents=True, exist_ok=True)     # only if dataset is not present 
        print(f"Downloading {FILENAME} to {data_dir} ...")
        urllib.request.urlretrieve(URL, path)           # the download 
    if _md5(path) != MD5:                               # compare checksum to the expected checksum 
        raise RuntimeError(f"{path} has the wrong checksum. Delete it and download again from {URL}")
    return path                                         # return file path 



def load_train_val(flatten=True, scale=True):
    """
    Training and validation data, for all tuning and model selection.
    Returns X_train, y_train, X_val, y_val.

    flatten=True  -> X has shape (n, 784): FFNN, logistic regression, trees.
    flatten=False -> X has shape (n, 28, 28): CNN.
    scale=True    -> pixel values divided by 255.
    """
    path = download_organamnist()

    with np.load(path) as data:
        X_train = data["train_images"]
        y_train = data["train_labels"].ravel()

        X_val = data["val_images"]
        y_val = data["val_labels"].ravel()

    X_train = X_train.astype(np.float64)
    X_val = X_val.astype(np.float64)

    if scale:
        X_train = X_train / 255.0
        X_val = X_val / 255.0

    if flatten:
        X_train = X_train.reshape(len(X_train), -1)
        X_val = X_val.reshape(len(X_val), -1)

    return X_train, y_train.astype(int), X_val, y_val.astype(int)


def load_test(flatten=True, scale=True):
    """
    Load test data for final model evaluation.
    Returns X_test, y_test.
    Same shapes as load_train_val for flatten and scale. 
    """
    path = download_organamnist()

    with np.load(path) as data:
        X_test = data["test_images"]
        y_test = data["test_labels"].ravel()

    X_test = X_test.astype(np.float64)

    if scale:
        X_test = X_test / 255.0

    if flatten:
        X_test = X_test.reshape(len(X_test), -1)

    return X_test, y_test.astype(int)


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
