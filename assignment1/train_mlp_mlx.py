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
This module implements training and evaluation of a multi-layer perceptron in MLX.
MLX is Apple's machine learning framework optimized for Apple Silicon.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import mlx.core as mx
import numpy as np
import os
from tqdm.auto import tqdm
from copy import deepcopy
import cifar10_utils

import torch


class LinearModule:
    """Linear module in MLX."""
    
    def __init__(self, in_features, out_features, input_layer=False):
        """Initialize linear layer with Kaiming initialization."""
        self.in_features = in_features
        self.out_features = out_features
        
        # Kaiming initialization
        if input_layer:
            std = np.sqrt(1.0 / in_features)
        else:
            std = np.sqrt(2.0 / in_features)
        
        self.params = {
            'weight': mx.array(np.random.randn(out_features, in_features) * std),
            'bias': mx.zeros(out_features)
        }
        self.grads = {
            'weight': mx.zeros((out_features, in_features)),
            'bias': mx.zeros(out_features)
        }
        self.x = None
    
    def forward(self, x):
        """Forward pass."""
        self.x = x
        out = x @ self.params['weight'].T + self.params['bias']
        return out
    
    def backward(self, dout):
        """Backward pass."""
        dx = dout @ self.params['weight']
        self.grads['weight'] = dout.T @ self.x
        self.grads['bias'] = mx.sum(dout, axis=0)
        return dx
    
    def clear_cache(self):
        """Clear cached tensors."""
        self.x = None


class ELUModule:
    """ELU activation module in MLX."""
    
    def __init__(self, alpha=1.0):
        self.alpha = alpha
        self.x = None
    
    def forward(self, x):
        """Forward pass."""
        self.x = x
        # MLX doesn't have a built-in ELU, so we implement it
        out = mx.where(x >= 0, x, self.alpha * (mx.exp(x) - 1))
        return out
    
    def backward(self, dout):
        """Backward pass."""
        dx = dout * mx.where(self.x >= 0, 1, self.alpha * mx.exp(self.x))
        return dx
    
    def clear_cache(self):
        """Clear cached tensors."""
        self.x = None


class SoftMaxModule:
    """Softmax activation module in MLX."""
    
    def __init__(self):
        self.out = None
    
    def forward(self, x):
        """Forward pass with numerical stability."""
        max_x = mx.max(x, axis=1, keepdims=True)
        stabilized_x = x - max_x
        exp_x = mx.exp(stabilized_x)
        sum_exp_x = mx.sum(exp_x, axis=1, keepdims=True)
        out = exp_x / sum_exp_x
        self.out = out
        return out
    
    def backward(self, dout):
        """Backward pass."""
        prod = self.out * dout
        sum_prod = mx.sum(prod, axis=1, keepdims=True)
        dx = self.out * (dout - sum_prod)
        return dx
    
    def clear_cache(self):
        """Clear cached tensors."""
        self.out = None


class CrossEntropyModule:
    """Cross entropy loss module in MLX."""
    
    def forward(self, x, y):
        """Forward pass."""
        S = x.shape[0]
        # Clip for numerical stability
        x_clipped = mx.clip(x, 1e-15, 1.0)
        # Create indices for gathering
        indices = mx.arange(S)
        log_probs = -mx.log(x_clipped[indices, y])
        out = mx.mean(log_probs)
        return out
    
    def backward(self, x, y):
        """Backward pass."""
        S = x.shape[0]
        indices = mx.arange(S)
        x_clipped = mx.clip(x, 1e-15, 1.0)
        
        # Create gradient array
        dx = mx.zeros_like(x)
        vals = (-1.0 / x_clipped[indices, y]) / S
        
        # MLX doesn't support advanced indexing for assignment like NumPy
        # So we'll work around this by creating the gradient differently
        y_one_hot = mx.zeros((S, x.shape[1]))
        # Build gradient manually
        dx_np = np.zeros(x.shape)
        dx_np[np.arange(S), np.array(y)] = np.array(vals)
        dx = mx.array(dx_np)
        
        return dx


class MLP:
    """Multi-layer perceptron in MLX."""
    
    def __init__(self, n_inputs, n_hidden, n_classes):
        """Initialize MLP with specified architecture."""
        self.layers = []
        
        # First layer with activation
        self.layers.append(LinearModule(n_inputs, n_hidden[0], input_layer=True))
        self.layers.append(ELUModule(alpha=1.0))
        
        # Hidden layers with activation
        for i in range(1, len(n_hidden)):
            self.layers.append(LinearModule(n_hidden[i-1], n_hidden[i]))
            self.layers.append(ELUModule(alpha=1.0))
        
        # Output layer
        self.layers.append(LinearModule(n_hidden[-1], n_classes))
        
        # Softmax
        self.layers.append(SoftMaxModule())
    
    def forward(self, x):
        """Forward pass through all layers."""
        out = x
        for layer in self.layers:
            out = layer.forward(out)
        return out
    
    def backward(self, dout):
        """Backward pass through all layers."""
        for layer in reversed(self.layers):
            dout = layer.backward(dout)
    
    def clear_cache(self):
        """Clear caches from all layers."""
        for layer in self.layers:
            layer.clear_cache()


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


def evaluate_model(model, data_loader):
    """
    Performs evaluation of the MLP model on a given dataset.
    
    Args:
        model: An instance of 'MLP', the model to evaluate
        data_loader: The data loader of the dataset to evaluate
    
    Returns:
        avg_accuracy: scalar float, average accuracy on the dataset
    """
    total_correct = 0
    total_samples = 0
    
    for images, labels in data_loader:
        # Flatten images from (batch_size, C, H, W) to (batch_size, C*H*W)
        images_flat = images.reshape(images.shape[0], -1)
        
        # Convert to MLX arrays
        images_mlx = mx.array(images_flat)
        labels_mlx = mx.array(labels)
        
        # Forward pass
        outputs = model.forward(images_mlx)
        preds = mx.argmax(outputs, axis=1)
        
        # Compute accuracy
        total_correct += float(mx.sum(preds == labels_mlx))
        total_samples += labels.shape[0]
    
    avg_accuracy = total_correct / total_samples
    return avg_accuracy


def train(hidden_dims, lr, batch_size, epochs, seed, data_dir):
    """
    Performs a full training cycle of MLP model.
    
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
    
    # Initialize model and loss module
    from functools import reduce
    flatten_size = reduce(lambda x, y: x * y, cifar10['test'][0][0].shape)
    model = MLP(n_inputs=flatten_size, n_hidden=hidden_dims, n_classes=10)
    loss_module = CrossEntropyModule()
    
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
        
        for images, labels in tqdm(cifar10_loader['train'], desc=f'Epoch {epoch+1}/{epochs}'):
            # Flatten images
            images_flat = images.reshape(images.shape[0], -1)
            
            # Convert to MLX arrays
            images_mlx = mx.array(images_flat)
            labels_mlx = mx.array(labels)
            
            # Forward pass
            predictions = model.forward(images_mlx)
            loss = loss_module.forward(predictions, labels_mlx)
            
            # Backward pass
            dout = loss_module.backward(predictions, labels_mlx)
            model.backward(dout)
            
            # Update parameters using SGD
            for layer in model.layers:
                if hasattr(layer, 'params') and layer.params.get('weight') is not None:
                    layer.params['weight'] = layer.params['weight'] - lr * layer.grads['weight']
                    layer.params['bias'] = layer.params['bias'] - lr * layer.grads['bias']
            
            # Evaluate MLX operations to ensure they execute
            mx.eval(loss)
            for layer in model.layers:
                if hasattr(layer, 'params') and layer.params.get('weight') is not None:
                    mx.eval(layer.params['weight'])
                    mx.eval(layer.params['bias'])
            
            # Track metrics
            epoch_losses.append(float(loss))
            batch_accuracy = accuracy(predictions, labels_mlx)
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
        
        # Save best model
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            best_model = deepcopy(model)
    
    # Test best model
    test_accuracy = evaluate_model(best_model, cifar10_loader['test'])
    print(f'\nTest Accuracy: {test_accuracy:.4f}')
    
    # Logging dictionary
    logging_dict = {
        'train_losses': train_losses,
        'train_accuracies': train_accuracies,
        'val_accuracies': val_accuracies,
        'test_accuracy': test_accuracy,
        'best_val_accuracy': best_val_accuracy
    }
    
    return best_model, val_accuracies, test_accuracy, logging_dict


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
