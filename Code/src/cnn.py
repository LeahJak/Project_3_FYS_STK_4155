# Convolutional neural network in PyTorch.
# Owner: CNN person.
#
# Plan (fill in):
# - Input: src.data.load_train_val(flatten=False) gives (n, 28, 28);
#   convert with torch.tensor(X, dtype=torch.float32).unsqueeze(1) -> (n, 1, 28, 28).
# - Model: a few Conv2d -> ReLU -> MaxPool2d blocks, then Linear layers with
#   11 outputs. Use nn.CrossEntropyLoss (it applies the softmax itself, so
#   apply torch.softmax to the outputs before saving probabilities).
# - Training loop with Adam, minibatches, early stopping on the validation set.
# - Do NOT use horizontal flips as augmentation: they turn left organs into
#   right organs (femur, kidney, lung) while keeping the label.
# - Hyperparameters: learning rate, number of filters, kernel size, depth,
#   dropout / weight decay. Heat maps or curves with src.plotting.
# - Number of parameters: sum(p.numel() for p in model.parameters() if p.requires_grad).
# - Final run: probabilities on the full test set -> src.evaluation.save_results(
#       "cnn", proba, y_test, train_time, n_params).
# - Figures: learned filters of the first layer and a few feature maps.
#
# Seeds: torch.manual_seed(seed) and np.random.default_rng(seed), seed from src.data.
