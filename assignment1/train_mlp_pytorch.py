################################################################################
# MIT License
#
# Copyright (c) 2025 University of Amsterdam
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to conditions.
#
# Author: Deep Learning Course (UvA) | Fall 2025
# Date Created: 2025-10-28
################################################################################
"""
This module implements training and evaluation of a multi-layer perceptron in PyTorch.
You should fill in code into indicated sections.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import numpy as np
import os
from copy import deepcopy
from tqdm.auto import tqdm
from mlp_pytorch import MLP
import cifar10_utils

import torch
import torch.nn as nn
import torch.optim as optim


def accuracy(predictions, targets):
    """
    Computes the prediction accuracy, i.e. the average of correct predictions
    of the network.

    Args:
      predictions: 2D float array of size [batch_size, n_classes], predictions of the model (logits)
      llabels: 1D int array of size [batch_size]. Ground truth labels for
               each sample in the batch
    Returns:
      accuracy: scalar float, the accuracy of predictions,
                i.e. the average correct predictions over the whole batch

    TODO:
    Implement accuracy computation.
    """

    #######################
    # PUT YOUR CODE HERE  #
    #######################
    pred_labels = torch.argmax(predictions, dim=1)
    correct = (pred_labels == targets).sum().item()
    accuracy = correct / targets.size(0)
    #######################
    # END OF YOUR CODE    #
    #######################

    return accuracy


def evaluate_model(model, data_loader):
    """
    Performs the evaluation of the MLP model on a given dataset.

    Args:
      model: An instance of 'MLP', the model to evaluate.
      data_loader: The data loader of the dataset to evaluate.
    Returns:
      avg_accuracy: scalar float, the average accuracy of the model on the dataset.

    TODO:
    Implement evaluation of the MLP model on a given dataset.

    Hint: make sure to return the average accuracy of the whole dataset,
          independent of batch sizes (not all batches might be the same size).
    """

    #######################
    # PUT YOUR CODE HERE  #
    #######################
    model.eval()
    total_accuracy = 0.0
    total_samples = 0
    with torch.no_grad():
        for inputs, targets in data_loader:
            inputs, targets = inputs.to(model.device), targets.to(model.device)
            inputs = inputs.view(inputs.size(0), -1)  # Flatten the input
            # print(inputs.shape, targets.shape)
            outputs = model(inputs)
            batch_size = targets.size(0)
            total_accuracy += accuracy(outputs, targets) * batch_size
            total_samples += batch_size
    avg_accuracy = total_accuracy / total_samples
    #######################
    # END OF YOUR CODE    #
    #######################

    return avg_accuracy


def train(hidden_dims, lr, use_batch_norm, batch_size, epochs, seed, data_dir):
    """
    Performs a full training cycle of MLP model.

    Args:
      hidden_dims: A list of ints, specificying the hidden dimensionalities to use in the MLP.
      lr: Learning rate of the SGD to apply.
      use_batch_norm: If True, adds batch normalization layer into the network.
      batch_size: Minibatch size for the data loaders.
      epochs: Number of training epochs to perform.
      seed: Seed to use for reproducible results.
      data_dir: Directory where to store/find the CIFAR10 dataset.
    Returns:
      model: An instance of 'MLP', the trained model that performed best on the validation set.
      val_accuracies: A list of scalar floats, containing the accuracies of the model on the
                      validation set per epoch (element 0 - performance after epoch 1)
      test_accuracy: scalar float, average accuracy on the test dataset of the model that
                     performed best on the validation.
      logging_dict: An arbitrary object containing logging information. This is for you to
                    decide what to put in here.

    TODO:
    - Implement the training of the MLP model.
    - Evaluate your model on the whole validation set each epoch.
    - After finishing training, evaluate your model that performed best on the validation set,
      on the whole test dataset.
    - Integrate _all_ input arguments of this function in your training. You are allowed to add
      additional input argument if you assign it a default value that represents the plain training
      (e.g. '..., new_param=False')

    Hint: you can save your best model by deepcopy-ing it.
    """

    # Set the random seeds for reproducibility
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():  # GPU operation have separate seed
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.determinstic = True
        torch.backends.cudnn.benchmark = False

    # Set default device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Loading the dataset
    cifar10 = cifar10_utils.get_cifar10(data_dir)
    cifar10_loader = cifar10_utils.get_dataloader(
        cifar10, batch_size=batch_size, return_numpy=False
    )

    #######################
    # PUT YOUR CODE HERE  #
    #######################
    from functools import reduce

    n_inputs = reduce(lambda x, y: x * y, cifar10["test"][0][0].shape)
    n_classes = 10
    # TODO: Initialize model and loss module
    model = MLP(n_inputs, hidden_dims, n_classes, use_batch_norm=use_batch_norm).to(
        device
    )
    loss_module = nn.CrossEntropyLoss()
    # TODO: Training loop including validation
    optimizer = optim.SGD(model.parameters(), lr=lr)
    best_val_accuracy = 0.0
    best_model = None
    train_accuracies = []
    val_accuracies = []
    train_losses = []
    for epoch in range(epochs):
        model.train()
        for inputs, targets in tqdm(cifar10_loader["train"]):
            inputs, targets = inputs.to(device), targets.to(device)
            inputs = inputs.view(inputs.size(0), -1)  # Flatten the input
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = loss_module(outputs, targets)
            loss.backward()
            optimizer.step()
        train_accuracy = evaluate_model(model, cifar10_loader["train"])
        val_accuracy = evaluate_model(model, cifar10_loader["validation"])
        train_accuracies.append(train_accuracy)
        val_accuracies.append(val_accuracy)
        train_losses.append(loss.item())
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            best_model = deepcopy(model)
        print(
            f"Epoch {epoch+1}/{epochs}, Train Accuracy: {train_accuracy:.4f}, Val Accuracy: {val_accuracy:.4f}"
        )
    model = best_model
    #######################
    # TODO: Do optimization with the simple SGD optimizer
    # val_accuracies = []
    # TODO: Test best model
    test_accuracy = evaluate_model(model, cifar10_loader["test"])
    # TODO: Add any information you might want to save for plotting
    logging_dict = {
        "train_losses": train_losses,
        "train_accuracies": train_accuracies,
        "val_accuracies": val_accuracies,
        "test_accuracy": test_accuracy,
    }
    print(
        f"Best Val Accuracy: {best_val_accuracy:.4f}, Test Accuracy of Best Model: {test_accuracy:.4f}"
    )
    #######################
    # END OF YOUR CODE    #
    #######################

    return model, val_accuracies, test_accuracy, logging_dict


if __name__ == "__main__":
  # Command line arguments
  parser = argparse.ArgumentParser()

  # Model hyperparameters
  parser.add_argument(
    "--hidden_dims",
    default=[128],
    type=int,
    nargs="+",
    help='Hidden dimensionalities to use inside the network. To specify multiple, use " " to separate them. Example: "256 128"',
  )
  parser.add_argument(
    "--use_batch_norm",
    action="store_true",
    help="Use this option to add Batch Normalization layers to the MLP.",
  )

  # Optimizer hyperparameters
  parser.add_argument("--lr", default=0.1, type=float, help="Learning rate to use")
  parser.add_argument("--batch_size", default=128, type=int, help="Minibatch size")

  # Other hyperparameters
  parser.add_argument("--epochs", default=10, type=int, help="Max number of epochs")
  parser.add_argument(
    "--seed", default=42, type=int, help="Seed to use for reproducing results"
  )
  parser.add_argument(
    "--data_dir",
    default="data/",
    type=str,
    help="Data directory where to store/find the CIFAR10 dataset.",
  )

  args = parser.parse_args()
  kwargs = vars(args)

  # Build parameter string for filenames
  import os, re, json
  hidden_str = "-".join(map(str, args.hidden_dims))
  data_dir_name = os.path.basename(os.path.normpath(args.data_dir))
  param_str = (
    f"hd={hidden_str}"
    f"_bn={int(args.use_batch_norm)}"
    f"_lr={args.lr}"
    f"_bs={args.batch_size}"
    f"_ep={args.epochs}"
    f"_seed={args.seed}"
    f"_dd={data_dir_name}"
  )
  # Sanitize (remove spaces, problematic chars)
  param_str = re.sub(r"[^\w=.-]", "_", param_str)

  # Print hyperparameters
  print("=" * 50)
  print("Training with the following hyperparameters:")
  print("=" * 50)
  for key, value in kwargs.items():
    print(f"{key}: {value}")
  print("=" * 50)

  model, val_accuracies, test_accuracy, logging_dict = train(**kwargs)

  # Optional: also store full args JSON
  logging_dict["args"] = kwargs

  import matplotlib.pyplot as plt

  # Loss curve
  plt.figure(figsize=(10, 6), dpi=300)
  plt.plot(logging_dict["train_losses"], label="Train Loss")
  plt.xlabel("Epoch")
  plt.ylabel("Loss")
  plt.title("Training Loss Curve")
  plt.legend()
  loss_path = f"loss_torch_{param_str}.png"
  plt.savefig(loss_path, dpi=300, bbox_inches="tight")
  print(f"Saved loss plot to {loss_path}")

  # Accuracy curves
  plt.figure(figsize=(10, 6), dpi=300)
  plt.plot(logging_dict["train_accuracies"], label="Train Accuracy")
  plt.plot(logging_dict["val_accuracies"], label="Validation Accuracy")
  plt.axhline(
    y=logging_dict["test_accuracy"],
    color="r",
    linestyle="--",
    label="Test Accuracy",
  )
  plt.xlabel("Epoch")
  plt.ylabel("Accuracy")
  plt.title("Train and Validation Accuracy")
  plt.legend()
  acc_path = f"acc_torch_{param_str}.png"
  plt.savefig(acc_path, dpi=300, bbox_inches="tight")
  print(f"Saved accuracy plot to {acc_path}")

  # Save model (state_dict preferred)
  import pickle
  model_path = f"model_torch_{param_str}.pkl"
  with open(model_path, "wb") as f:
    pickle.dump(model, f)
  print(f"Model saved to {model_path}")

  # Save metrics
  metrics_path = f"metrics_torch_{param_str}.pkl"
  with open(metrics_path, "wb") as f:
    pickle.dump(logging_dict, f)
  print(f"Metrics saved to {metrics_path}")

  # Also save a lightweight JSON summary
  summary = {
    "best_val_accuracy": max(val_accuracies) if val_accuracies else None,
    "test_accuracy": test_accuracy,
    "params": kwargs,
    "filenames": {
      "loss": loss_path,
      "accuracy": acc_path,
      "model": model_path,
      "metrics": metrics_path,
    },
  }
  with open(f"summary_{param_str}.json", "w") as f:
    json.dump(summary, f, indent=2)
  print(f"Summary saved to summary_{param_str}.json")
