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
This module implements training and evaluation of a multi-layer perceptron in JAX
using Flax neural network library.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import jax
import jax.numpy as jnp
from jax import random, jit, grad, value_and_grad
import flax.linen as nn
import optax
import numpy as np
import os
from tqdm.auto import tqdm
from functools import partial
import cifar10_utils

import torch


class MLP(nn.Module):
    """Multi-layer perceptron using Flax."""
    
    n_hidden: list
    n_classes: int
    
    def setup(self):
        """Initialize layers."""
        # Build layers dynamically
        self.layers = []
        
        # Input and hidden layers with ELU activation
        for i, hidden_dim in enumerate(self.n_hidden):
            # Use custom initializer for Kaiming initialization
            if i == 0:
                # First layer (input layer)
                kernel_init = nn.initializers.variance_scaling(
                    scale=1.0, mode='fan_in', distribution='normal'
                )
            else:
                # Hidden layers (for ELU activation)
                kernel_init = nn.initializers.variance_scaling(
                    scale=2.0, mode='fan_in', distribution='normal'
                )
            
            self.layers.append(nn.Dense(
                hidden_dim,
                kernel_init=kernel_init,
                bias_init=nn.initializers.zeros
            ))
        
        # Output layer
        kernel_init = nn.initializers.variance_scaling(
            scale=2.0, mode='fan_in', distribution='normal'
        )
        self.output_layer = nn.Dense(
            self.n_classes,
            kernel_init=kernel_init,
            bias_init=nn.initializers.zeros
        )
    
    def __call__(self, x):
        """Forward pass through all layers."""
        # Pass through hidden layers with ELU activation
        for layer in self.layers:
            x = layer(x)
            x = nn.elu(x, alpha=1.0)
        
        # Output layer (no activation)
        x = self.output_layer(x)
        return x


def cross_entropy_loss(logits, labels):
    """
    Compute cross entropy loss from logits.
    
    Args:
        logits: Model predictions of shape [batch_size, n_classes]
        labels: Ground truth labels of shape [batch_size]
    
    Returns:
        loss: Scalar loss value
    """
    # Convert labels to one-hot
    one_hot_labels = jax.nn.one_hot(labels, logits.shape[-1])
    # Compute softmax cross entropy
    log_probs = jax.nn.log_softmax(logits)
    loss = -jnp.mean(jnp.sum(one_hot_labels * log_probs, axis=-1))
    return loss


def accuracy(logits, labels):
    """
    Computes the prediction accuracy.
    
    Args:
        logits: JAX array of shape [batch_size, n_classes], model predictions (logits)
        labels: JAX array of shape [batch_size], ground truth labels
    
    Returns:
        accuracy: scalar float, the accuracy between 0 and 1
    """
    predictions = jnp.argmax(logits, axis=-1)
    return jnp.mean(predictions == labels)


def preload_data_to_device(data_loader):
    """
    Preload all data batches to JAX arrays (GPU memory).
    
    Args:
        data_loader: Data loader with numpy arrays
    
    Returns:
        List of (images, labels) tuples as JAX arrays
    """
    preloaded_batches = []
    for images, labels in data_loader:
        # Flatten and convert to JAX arrays
        images_jax = jnp.array(images.reshape(images.shape[0], -1))
        labels_jax = jnp.array(labels)
        preloaded_batches.append((images_jax, labels_jax))
    return preloaded_batches


@jit
def train_step(state, batch, optimizer):
    """
    Perform a single training step.
    
    Args:
        state: Current model parameters
        batch: Tuple of (images, labels)
        optimizer: Optimizer state
    
    Returns:
        Updated state, loss, and accuracy
    """
    images, labels = batch
    
    def loss_fn(params):
        logits = state.apply_fn({'params': params}, images)
        loss = cross_entropy_loss(logits, labels)
        return loss, logits
    
    # Compute gradients
    grad_fn = value_and_grad(loss_fn, has_aux=True)
    (loss, logits), grads = grad_fn(state.params)
    
    # Update parameters
    updates, optimizer = state.tx.update(grads, optimizer, state.params)
    params = optax.apply_updates(state.params, updates)
    
    # Compute accuracy
    acc = accuracy(logits, labels)
    
    return state.replace(params=params), optimizer, loss, acc


@jit
def eval_step(state, batch):
    """
    Perform a single evaluation step.
    
    Args:
        state: Current model parameters
        batch: Tuple of (images, labels)
    
    Returns:
        Accuracy for the batch
    """
    images, labels = batch
    logits = state.apply_fn({'params': state.params}, images)
    return accuracy(logits, labels)


def evaluate_model(state, preloaded_data):
    """
    Performs evaluation of the MLP model on preloaded dataset.
    
    Args:
        state: Model state with parameters
        preloaded_data: List of preloaded (images, labels) tuples as JAX arrays
    
    Returns:
        avg_accuracy: scalar float, average accuracy on the dataset
    """
    accuracies = []
    
    for batch in preloaded_data:
        batch_acc = eval_step(state, batch)
        accuracies.append(float(batch_acc))
    
    avg_accuracy = np.mean(accuracies)
    return avg_accuracy


class TrainState:
    """Simple container for training state."""
    
    def __init__(self, params, apply_fn, tx):
        self.params = params
        self.apply_fn = apply_fn
        self.tx = tx
    
    def replace(self, **kwargs):
        """Create a new state with updated fields."""
        new_state = TrainState(self.params, self.apply_fn, self.tx)
        for key, value in kwargs.items():
            setattr(new_state, key, value)
        return new_state


def train(hidden_dims, lr, batch_size, epochs, seed, data_dir):
    """
    Performs a full training cycle of MLP model using JAX/Flax.
    
    Args:
        hidden_dims: A list of ints, specifying the hidden dimensionalities
        lr: Learning rate of the SGD
        batch_size: Minibatch size for the data loaders
        epochs: Number of training epochs
        seed: Seed for reproducible results
        data_dir: Directory where to store/find the CIFAR10 dataset
    
    Returns:
        state: Trained model state
        val_accuracies: List of validation accuracies per epoch
        test_accuracy: Test accuracy of best model
        logging_dict: Dictionary containing training logs
    """
    
    # Set random seeds for reproducibility
    np.random.seed(seed)
    torch.manual_seed(seed)
    key = random.PRNGKey(seed)
    
    # Load dataset
    cifar10 = cifar10_utils.get_cifar10(data_dir)
    cifar10_loader = cifar10_utils.get_dataloader(
        cifar10, batch_size=batch_size, return_numpy=True
    )
    
    # Get input dimensions
    from functools import reduce
    flatten_size = reduce(lambda x, y: x * y, cifar10['test'][0][0].shape)
    
    # Preload all data to device
    print("Preloading data to device...")
    train_data = preload_data_to_device(cifar10_loader['train'])
    val_data = preload_data_to_device(cifar10_loader['validation'])
    test_data = preload_data_to_device(cifar10_loader['test'])
    print(f"Preloaded {len(train_data)} training batches, {len(val_data)} validation batches, {len(test_data)} test batches")
    
    # Initialize model
    model = MLP(n_hidden=hidden_dims, n_classes=10)
    
    # Initialize parameters
    key, init_key = random.split(key)
    dummy_input = jnp.ones((1, flatten_size))
    variables = model.init(init_key, dummy_input)
    params = variables['params']
    
    # Initialize optimizer
    tx = optax.sgd(learning_rate=lr)
    opt_state = tx.init(params)
    
    # Create training state
    state = TrainState(params=params, apply_fn=model.apply, tx=tx)
    
    # Training setup
    val_accuracies = []
    train_losses = []
    train_accuracies = []
    best_val_accuracy = 0.0
    best_params = None
    
    # Training loop
    for epoch in range(epochs):
        # Training phase
        epoch_losses = []
        epoch_train_accuracies = []
        
        for batch in tqdm(train_data, desc=f'Epoch {epoch+1}/{epochs}', leave=False):
            # Perform training step
            state, opt_state, loss, acc = train_step(state, batch, opt_state)
            
            # Store metrics
            epoch_losses.append(float(loss))
            epoch_train_accuracies.append(float(acc))
        
        # Calculate average training metrics
        avg_train_loss = np.mean(epoch_losses)
        avg_train_accuracy = np.mean(epoch_train_accuracies)
        train_losses.append(avg_train_loss)
        train_accuracies.append(avg_train_accuracy)
        
        # Validation phase
        val_accuracy = evaluate_model(state, val_data)
        val_accuracies.append(val_accuracy)
        
        print(f'Epoch {epoch+1}/{epochs} - Train Loss: {avg_train_loss:.4f}, '
              f'Train Acc: {avg_train_accuracy:.4f}, Val Acc: {val_accuracy:.4f}')
        
        # Save best model parameters
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            best_params = jax.tree_map(lambda x: x.copy(), state.params)
    
    # Restore best model parameters
    state = state.replace(params=best_params)
    
    # Test best model
    test_accuracy = evaluate_model(state, test_data)
    print(f'\nTest Accuracy: {test_accuracy:.4f}')
    
    # Logging dictionary
    logging_dict = {
        'train_losses': train_losses,
        'train_accuracies': train_accuracies,
        'val_accuracies': val_accuracies,
        'test_accuracy': test_accuracy,
        'best_val_accuracy': best_val_accuracy
    }
    
    return state, val_accuracies, test_accuracy, logging_dict


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
