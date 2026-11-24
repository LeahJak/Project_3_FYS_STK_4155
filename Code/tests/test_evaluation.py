# Tests of src/evaluation.py.
#
# Skeleton: test names and docstrings suggested by an LLM (Claude, November
# 2026); the tests are to be written by the group. Remove pytest.skip(...)
# when a test is written.
#
# Tip: pass pytest's built-in tmp_path as results_dir / configs_dir, so the
# tests never write into the real results/ and configs/ folders. Fake
# probabilities: softmax of random numbers, shape (N_SAMPLES["test"], N_CLASSES).

import numpy as np
import pytest

from src.data import N_CLASSES, N_SAMPLES
from src.evaluation import save_results, load_results, load_all_results, save_config, load_config


def test_roundtrip():
    """save_results followed by load_results gives back the same proba, y_test, train_time, n_params and notes."""
    pytest.skip("not written yet")


def test_rejects_bad_input():
    """save_results refuses rows not summing to 1, part of the test set, and the wrong number of classes."""
    pytest.skip("not written yet")


def test_load_all_checks_same_labels():
    """load_all_results loads every method, and raises an error if one method has reordered y_test."""
    pytest.skip("not written yet")


def test_config_roundtrip():
    """save_config followed by load_config gives back the same dict."""
    pytest.skip("not written yet")
