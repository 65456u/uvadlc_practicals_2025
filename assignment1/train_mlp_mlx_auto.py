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
# Date Created: 2025-11-04
################################################################################
"""
This module implements training and evaluation of a multi-layer perceptron in MLX
using built-in modules from mlx.nn instead of custom implementations.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np
import os
from tqdm.auto import tqdm
import cifar10_utils

import torch


class MLP(nn.Module):
    """Multi-layer perceptron using MLX built-in modules."""
    
    def __init__(self, n_inputs, n_hidden, n_classes):
        """
        Initialize MLP with specified architecture using built-in modules.
        
        Args:
            n_inputs: Number of input features
            n_hidden: List of hidden layer dimensions
            n_classes: Number of output classes
        """
        super().__init__()
        
        # Build layers list
        layers = []
        
        # Input layer with Kaiming initialization
        input_layer = nn.Linear(n_inputs, n_hidden[0])
        self._init_weights(input_layer, n_inputs, is_input_layer=True)
        layers.append(input_layer)
        layers.append(nn.ELU(alpha=1.0))
        
        # Hidden layers with Kaiming initialization
        for i in range(1, len(n_hidden)):
            hidden_layer = nn.Linear(n_hidden[i-1], n_hidden[i])
            self._init_weights(hidden_layer, n_hidden[i-1], is_input_layer=False)
            layers.append(hidden_layer)
            layers.append(nn.ELU(alpha=1.0))
        
        # Output layer with Kaiming initialization
        output_layer = nn.Linear(n_hidden[-1], n_classes)
        self._init_weights(output_layer, n_hidden[-1], is_input_layer=False)
        layers.append(output_layer)
        
        # Store as sequential
        self.layers = layers
    
    def _init_weights(self, layer, in_features, is_input_layer=False):
        """
        Initialize weights using Kaiming initialization.
        
        Args:
            layer: Linear layer to initialize
            in_features: Number of input features
            is_input_layer: Whether this is the first layer after input
        """
        # Kaiming initialization
        if is_input_layer:
            std = np.sqrt(1.0 / in_features)
        else:
            std = np.sqrt(2.0 / in_features)
        
        # Initialize weights
        layer.weight = mx.array(np.random.randn(layer.weight.shape[0], layer.weight.shape[1]) * std)
        # Initialize biases to zero
        layer.bias = mx.zeros_like(layer.bias)
    
    def __call__(self, x):
        """Forward pass through all layers."""
        for layer in self.layers:
            x = layer(x)
        return x


def cross_entropy_loss(logits, targets):
    """
    Compute cross entropy loss from logits.
    
    Args:
        logits: Model predictions of shape [batch_size, n_classes]
        targets: Ground truth labels of shape [batch_size]
    
    Returns:
        loss: Scalar loss value
    """
    # Apply log softmax for numerical stability
    log_probs = logits - mx.logsumexp(logits, axis=-1, keepdims=True)
    
    # Get log probabilities for target classes
    batch_size = logits.shape[0]
    target_log_probs = log_probs[mx.arange(batch_size), targets]
    
    # Return negative mean
    return -mx.mean(target_log_probs)


def accuracy(predictions, targets):
    """
    Computes the prediction accuracy.
    
    Args:
        predictions: MLX array of shape [batch_size, n_classes], model predictions (logits)
        targets: MLX array of shape [batch_size], ground truth labels
    
    Returns:
        accuracy: scalar float, the accuracy between 0 and 1
    """
    preds = mx.argmax(predictions, axis=1)
    correct = mx.sum(preds == targets)
    accuracy = float(correct) / predictions.shape[0]
    return accuracy


def loss_fn(model, X, y):
    """
    Compute loss for a batch.
    
    Args:
        model: MLP model
        X: Input batch
        y: Target labels
    
    Returns:
        loss: Scalar loss value
    """
    logits = model(X)
    return cross_entropy_loss(logits, y)


def evaluate_model(model, data_loader):
    """
    Performs evaluation of the MLP model on a given dataset.
    
    Args:
        model: An instance of 'MLP', the model to evaluate
        data_loader: The data loader of the dataset to evaluate
    
    Returns:
        avg_accuracy: scalar float, average accuracy on the dataset
    """
    accuracies = []
    
    for images, labels in data_loader:
        # Flatten images from (batch_size, C, H, W) to (batch_size, C*H*W)
        images_flat = images.reshape(images.shape[0], -1)
        
        # Convert to MLX arrays
        images_mlx = mx.array(images_flat)
        labels_mlx = mx.array(labels)
        
        # Forward pass
        outputs = model(images_mlx)
        
        # Compute accuracy for this batch
        batch_acc = accuracy(outputs, labels_mlx)
        accuracies.append(batch_acc)
    
    avg_accuracy = np.mean(accuracies)
    return avg_accuracy


def train(hidden_dims, lr, batch_size, epochs, seed, data_dir):
    """
    Performs a full training cycle of MLP model using MLX built-in modules.
    
    Args:
        hidden_dims: A list of ints, specifying the hidden dimensionalities
        lr: Learning rate of the SGD
        batch_size: Minibatch size for the data loaders
        epochs: Number of training epochs
        seed: Seed for reproducible results
        data_dir: Directory where to store/find the CIFAR10 dataset
    
    Returns:
        model: Trained MLP model
        val_accuracies: List of validation accuracies per epoch
        test_accuracy: Test accuracy of best model
        logging_dict: Dictionary containing training logs
    """
    
    # Set random seeds for reproducibility
    np.random.seed(seed)
    torch.manual_seed(seed)
    mx.random.seed(seed)
    
    # Load dataset
    cifar10 = cifar10_utils.get_cifar10(data_dir)
    cifar10_loader = cifar10_utils.get_dataloader(
        cifar10, batch_size=batch_size, return_numpy=True
    )
    
    # Initialize model
    from functools import reduce
    flatten_size = reduce(lambda x, y: x * y, cifar10['test'][0][0].shape)
    model = MLP(n_inputs=flatten_size, n_hidden=hidden_dims, n_classes=10)
    
    # Initialize optimizer (SGD)
    optimizer = optim.SGD(learning_rate=lr)
    
    # Create loss and gradient function
    loss_and_grad_fn = nn.value_and_grad(model, loss_fn)
    
    # Pre-compile the model by running a dummy forward pass
    dummy_input = mx.zeros((1, flatten_size))
    _ = model(dummy_input)
    mx.eval(model.parameters())
    
    # Training setup
    val_accuracies = []
    train_losses = []
    train_accuracies = []
    best_val_accuracy = 0.0
    best_model = None
    
    # Training loop
    for epoch in range(epochs):
        # Training phase
        epoch_losses = []
        epoch_train_accuracies = []
        
        for images, labels in tqdm(cifar10_loader['train'], desc=f'Epoch {epoch+1}/{epochs}', leave=False):
            # Flatten and convert to MLX arrays in one go
            batch_size = images.shape[0]
            images_mlx = mx.array(images.reshape(batch_size, -1))
            labels_mlx = mx.array(labels)
            
            # Compute loss and gradients
            loss, grads = loss_and_grad_fn(model, images_mlx, labels_mlx)
            
            # Update parameters
            optimizer.update(model, grads)
            
            # Evaluate parameters and loss in a single call
            mx.eval(model.parameters(), loss)
            
            # Store loss
            epoch_losses.append(float(loss))
            
            # Compute accuracy on the same batch
            logits = model(images_mlx)
            batch_accuracy = accuracy(logits, labels_mlx)
            epoch_train_accuracies.append(batch_accuracy)
        
        # Calculate average training metrics
        avg_train_loss = np.mean(epoch_losses)
        avg_train_accuracy = np.mean(epoch_train_accuracies)
        train_losses.append(avg_train_loss)
        train_accuracies.append(avg_train_accuracy)
        
        # Validation phase
        val_accuracy = evaluate_model(model, cifar10_loader['validation'])
        val_accuracies.append(val_accuracy)
        
        print(f'Epoch {epoch+1}/{epochs} - Train Loss: {avg_train_loss:.4f}, '
              f'Train Acc: {avg_train_accuracy:.4f}, Val Acc: {val_accuracy:.4f}')
        
        # Save best model (save parameters instead of deep copying the entire model)
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            # Store a copy of model parameters
            best_model_params = {k: v for k, v in model.parameters().items()}
            mx.eval(best_model_params)
    
    # Restore best model parameters
    model.update(best_model_params)
    
    # Test best model
    test_accuracy = evaluate_model(model, cifar10_loader['test'])
    print(f'\nTest Accuracy: {test_accuracy:.4f}')
    
    # Logging dictionary
    logging_dict = {
        'train_losses': train_losses,
        'train_accuracies': train_accuracies,
        'val_accuracies': val_accuracies,
        'test_accuracy': test_accuracy,
        'best_val_accuracy': best_val_accuracy
    }
    
    return model, val_accuracies, test_accuracy, logging_dict


if __name__ == '__main__':
    # Command line arguments
    parser = argparse.ArgumentParser()
    
    # Model hyperparameters
    parser.add_argument('--hidden_dims', default=[128], type=int, nargs='+',
                        help='Hidden dimensionalities to use inside the network. To specify multiple, use " " to separate them. Example: "256 128"')
    
    # Optimizer hyperparameters
    parser.add_argument('--lr', default=0.1, type=float,
                        help='Learning rate to use')
    parser.add_argument('--batch_size', default=128, type=int,
                        help='Minibatch size')

    # Other hyperparameters
    parser.add_argument('--epochs', default=10, type=int,
                        help='Max number of epochs')
    parser.add_argument('--seed', default=42, type=int,
                        help='Seed to use for reproducing results')
    parser.add_argument('--data_dir', default='data/', type=str,
                        help='Data directory where to store/find the CIFAR10 dataset.')
       
    args = parser.parse_args()
    kwargs = vars(args)
    
    train(**kwargs)
