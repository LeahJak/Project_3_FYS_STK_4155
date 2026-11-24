# Tests of src/data.py.
#
# Skeleton: test names and docstrings suggested by an LLM (Claude, November
# 2026); the tests are to be written by the group. Remove pytest.skip(...)
# when a test is written.
#
# Tip: the tests should not need the real 38 MB download. pytest's built-in
# tmp_path fixture gives a temporary folder where you can save a small fake
# organamnist.npz (same keys, a few random 28x28 uint8 images, labels of
# shape (n, 1)) and pass that folder as data_dir.

import numpy as np
import pytest

import src.data as data


def test_shapes_and_scaling():
    """load_train_val(): X has shape (n, 784), y is 1D integer, pixel values in [0, 1]."""
    pytest.skip("not written yet")


def test_unflattened_for_cnn():
    """flatten=False gives X of shape (n, 28, 28)."""
    pytest.skip("not written yet")


def test_labels_and_counts():
    """Labels are within 0-10, and class_counts adds up to the number of samples."""
    pytest.skip("not written yet")


def test_subset_balanced():
    """subset() gives n_per_class samples of each class, and each X stays paired with its own y."""
    pytest.skip("not written yet")


def test_missing_or_broken_file():
    """A missing (or, if you check MD5, damaged) data file gives a clear error."""
    pytest.skip("not written yet")
