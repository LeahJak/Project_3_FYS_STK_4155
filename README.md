# FYS-STK3155/4155 Project 3: Classifying organs in CT images

Group members: Nora Arnesen, Viktoria Krogen, Natalie Kristiansen, Leah Jakobsen

This repository contains the code and report for Project 3 (Path B: a classification study with several methods) in FYS-STK3155/4155, fall 2026. We classify 28×28 axial CT slices of 11 abdominal organs from **OrganAMNIST** (MedMNIST v2) with logistic regression, our own feed-forward neural network from Project 2, tree-based ensembles (random forest, gradient boosting) and a convolutional neural network, and compare them on the same test set and with the published literature.

## Data

OrganAMNIST is downloaded automatically (about 38 MB, from Zenodo) to `Code/data/` the first time `src/data.py` loads it. The data are not committed to the repository. If the download fails, download `organamnist.npz` from <https://zenodo.org/records/10519652> and put it in `Code/data/`.

We use the official split, which is by CT scan (115 scans for training, 16 for validation, 70 for testing): 34 561 training, 6 491 validation and 17 778 test images.

Cite:
- J. Yang et al., *MedMNIST v2 – A large-scale lightweight benchmark for 2D and 3D biomedical image classification*, Scientific Data **10**, 41 (2023).
- J. Yang, R. Shi, B. Ni, *MedMNIST Classification Decathlon: A Lightweight AutoML Benchmark for Medical Image Analysis*, IEEE ISBI (2021).
- P. Bilic et al., *The Liver Tumor Segmentation Benchmark (LiTS)*, Medical Image Analysis **84**, 102680 (2023).

## Repository layout

```
Project_3_FYS_STK_4155/
├── Code/
│   ├── src/                Shared code. Notebooks import from here.
│   │   ├── data.py             OrganAMNIST loading, class counts, subsets, P2 helpers
│   │   ├── network.py          FFNN + backprop (own code, from Project 2)
│   │   ├── functions.py        costs, activations, derivatives (from Project 2)
│   │   ├── optimizers.py       GD, SGD, RMSprop, Adam (from Project 2)
│   │   ├── plotting.py         shared plot style and helpers (from Project 2)
│   │   ├── metrics.py          P2 metrics + F1, ROC/AUC, McNemar
│   │   ├── trees.py            random forest, gradient boosting
│   │   ├── cnn.py              CNN in PyTorch
│   │   └── evaluation.py       shared format for test-set results and configs
│   ├── notebooks/          One notebook per method + data + comparison
│   ├── configs/            Chosen hyperparameters and seed per method (json)
│   ├── results/            Test-set probabilities per method (npz)
│   ├── tests/              pytest tests
│   ├── data/               Downloaded data (not committed)
│   ├── pyproject.toml
│   └── requirements.txt
├── figures/            All figures produced by the notebooks
└── Report/             PDF of the final report
```

## Installation

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r Code/requirements.txt
nbstripout --install          # strips notebook outputs on commit (once per clone)
```

## Reproducing the results

Run the notebooks in `Code/notebooks/` with **Kernel → Restart & Run All**:

| Notebook | Content | Produces |
|---|---|---|
| `01_data.ipynb` | Data, class balance, example images, preprocessing (part a) | figures |
| `02_logreg.ipynb` | Logistic regression baseline: tuning and final model | `results/logreg.npz`, `configs/logreg.json` |
| `03_ffnn.ipynb` | Own FFNN: tuning and final model | `results/ffnn.npz`, `configs/ffnn.json` |
| `04_trees.ipynb` | Random forest and gradient boosting | `results/random_forest.npz`, `results/gradient_boosting.npz` |
| `05_cnn.ipynb` | CNN (PyTorch), filters and feature maps | `results/cnn.npz`, `configs/cnn.json` |
| `06_comparison.ipynb` | All methods on the same test set, comparison with literature (part d) | figures, tables |

Notebooks 02–05 can be run in any order; 06 only reads the saved results, so it runs in seconds. All randomness uses the seed 2026 (`src.data.seed`).

## Running tests

From the `Code/` directory:

```bash
pytest -v
```

## Use of LLMs

LLMs were used to set up the repository structure, `src/data.py`, `src/evaluation.py`, the tests in `tests/test_data.py` and `tests/test_evaluation.py`, the notebook skeletons and the first draft of this README (Claude, November 2026). Every LLM-assisted function carries an `LLM-assisted` tag. For the full declaration, see the appendix of the report.
