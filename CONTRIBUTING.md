# Working in this repository

## Ownership
Each notebook has one owner, written at the top of the notebook. Only the owner edits it; this avoids merge conflicts in notebooks. Shared code in `Code/src/` can be edited by everyone, but tell the group first if you change a function others use.

## Branches
Work on a branch named after your part (`ffnn`, `trees`, `cnn`, `comparison`), and merge into `main` with a pull request when it runs from top to bottom.

## The test set
- Tune only on `load_train_val()` (validation split or cross-validation).
- `load_test()` is called once per method, in the final cell of its notebook, which saves the results with `save_results(...)` and the chosen hyperparameters with `save_config(...)`.
- After that cell has been run for the report, nothing is tuned further.

## Results format
`src/evaluation.save_results(method, proba, y_test, train_time, n_params, notes)`:
- `proba`: class probabilities of shape (17 778, 11) on the full test set, rows summing to 1;
- `train_time`: training time of the final model in seconds;
- `n_params`: number of trainable parameters (for trees, say in `notes` what you counted).

`06_comparison.ipynb` reads everything with `load_all_results()`.

## Figures
Save with `src.plotting.save_fig(name)` (PDF in `figures/`). Prefix names by notebook, e.g. `03_heatmap_eta_lambda`, `05_filters`.

## Seeds
Use `seed` from `src.data` everywhere: `np.random.default_rng(seed)`, `random_state=seed` in scikit-learn, `torch.manual_seed(seed)`.

## LLM use
Every LLM-assisted function or cell carries the inline `LLM-assisted` tag (tool, date, role) as in `src/data.py`, and must be tested and understood by you. Report text: LLMs only for spelling, grammar and minor phrasing (level 1).
