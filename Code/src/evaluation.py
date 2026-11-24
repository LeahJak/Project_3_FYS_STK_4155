# Shared format for final results and chosen hyperparameters.
#
# Every method ends its notebook with ONE final run:
#   1. train the chosen model,
#   2. predict class probabilities on the test set,
#   3. save_results(...) and save_config(...).
# notebooks/06_comparison.ipynb then loads everything with load_all_results().
#
# Method names: "logreg", "ffnn", "random_forest", "gradient_boosting", "cnn".
#
# Skeleton: function names and docstrings suggested by an LLM (Claude, November
# 2026); the function bodies are to be written by the group.

from pathlib import Path

import numpy as np

from src.data import N_CLASSES, N_SAMPLES

CODE_DIR = Path(__file__).resolve().parents[1]
RESULTS_DIR = CODE_DIR / "results"
CONFIGS_DIR = CODE_DIR / "configs"


def save_results(method, proba, y_test, train_time, n_params, notes="",
                 results_dir=RESULTS_DIR):
    """
    Save the test-set predictions of one method to results/<method>.npz.

    Parameters
    ----------
    method : str. Short name, e.g. "cnn".
    proba : array (n_test, 11). Predicted class probabilities on the full test set.
    y_test : array (n_test,). True labels, so 06 can check all methods used the same order.
    train_time : float. Training time of the final model in seconds.
    n_params : int. Number of trainable parameters (trees: say in notes what was counted).
    notes : str. Anything the comparison person should know.

    Suggestions:
    - Check that proba has shape (N_SAMPLES["test"], N_CLASSES) and rows sum to 1.
    - Save with np.savez_compressed.
    """
    raise NotImplementedError


def load_results(method, results_dir=RESULTS_DIR):
    """Load one method's results, e.g. as a dict with keys proba, y_test, train_time, n_params, notes."""
    raise NotImplementedError


def load_all_results(results_dir=RESULTS_DIR):
    """
    Load every results/*.npz, as {method: results}.

    Suggestion: check that all methods have identical y_test (same images, same order).
    """
    raise NotImplementedError


def save_config(method, config, configs_dir=CONFIGS_DIR):
    """
    Save the chosen hyperparameters and seed of one method to configs/<method>.json,
    e.g. {"seed": 2026, "eta": 1e-3, "lambda": 1e-4, "layers": [100]}.
    Hint: the json module (json.dump).
    """
    raise NotImplementedError


def load_config(method, configs_dir=CONFIGS_DIR):
    """Read configs/<method>.json back as a dict."""
    raise NotImplementedError
