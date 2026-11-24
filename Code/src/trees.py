# Tree-based ensembles: random forest and gradient boosting.
# Owner: trees person.
#
# Plan (fill in):
# - Random forest: sklearn.ensemble.RandomForestClassifier.
#   Hyperparameters: n_estimators, max_depth, max_features (features per split).
# - Gradient boosting: sklearn.ensemble.HistGradientBoostingClassifier
#   (fast on 34 561 x 784) and/or xgboost.XGBClassifier.
#   Hyperparameters: learning_rate, max_depth / max_leaf_nodes, number of trees.
# - Input: flattened images from src.data.load_train_val(flatten=True).
# - Tune on train/val (or cross-validation), heat maps with src.plotting.heatmap.
# - Final run: model.predict_proba(X_test) -> src.evaluation.save_results(
#       "random_forest" / "gradient_boosting", proba, y_test, train_time, n_params).
# - Feature importances: reshape model.feature_importances_ to (28, 28) and plot.
#
# Always pass random_state=seed (from src.data) for reproducibility.
