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
This module implements training and evaluation of a multi-layer perceptron in NumPy.
You should fill in code into indicated sections.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import numpy as np
import os
from tqdm.auto import tqdm
from copy import deepcopy
from mlp_numpy import MLP
from modules import CrossEntropyModule
import cifar10_utils

import torch


def accuracy(predictions, targets):
    """
    Computes the prediction accuracy, i.e. the average of correct predictions
    of the network.

    Args:
      predictions: 2D float array of size [batch_size, n_classes], predictions of the model (logits)
      labels: 1D int array of size [batch_size]. Ground truth labels for
              each sample in the batch
    Returns:
      accuracy: scalar float, the accuracy of predictions between 0 and 1,
                i.e. the average correct predictions over the whole batch

    TODO:
    Implement accuracy computation.
    """

    #######################
    # PUT YOUR CODE HERE  #
    #######################
    preds = np.argmax(predictions, axis=1)
    correct = np.sum(preds == targets)
    accuracy = correct / predictions.shape[0]
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
    total_correct = 0
    total_samples = 0
    
    for images, labels in data_loader:
        # Flatten images from (batch_size, C, H, W) to (batch_size, C*H*W)
        images_flat = images.reshape(images.shape[0], -1)
        outputs = model.forward(images_flat)
        preds = np.argmax(outputs, axis=1)
        total_correct += np.sum(preds == labels)
        total_samples += labels.shape[0]
    
    avg_accuracy = total_correct / total_samples
    #######################
    # END OF YOUR CODE    #
    #######################

    return avg_accuracy


def train(hidden_dims, lr, batch_size, epochs, seed, data_dir):
    """
    Performs a full training cycle of MLP model.

    Args:
      hidden_dims: A list of ints, specificying the hidden dimensionalities to use in the MLP.
      lr: Learning rate of the SGD to apply.
      batch_size: Minibatch size for the data loaders.
      epochs: Number of training epochs to perform.
      seed: Seed to use for reproducible results.
      data_dir: Directory where to store/find the CIFAR10 dataset.
    Returns:
      model: An instance of 'MLP', the trained model that performed best on the validation set.
      val_accuracies: A list of scalar floats, containing the accuracies of the model on the
                      validation set per epoch (element 0 - performance after epoch 1)
      test_accuracy: scalar float, average accuracy on the test dataset of the model that 
                     performed best on the validation. Between 0.0 and 1.0
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

    ## Loading the dataset
    cifar10 = cifar10_utils.get_cifar10(data_dir)
    cifar10_loader = cifar10_utils.get_dataloader(cifar10, batch_size=batch_size,
                                                  return_numpy=True)

    #######################
    # PUT YOUR CODE HERE  #
    #######################

    # TODO: Initialize model and loss module
    from functools import reduce
    flatten_size = reduce(lambda x, y: x * y, cifar10['test'][0][0].shape)
    model = MLP(n_inputs=flatten_size, n_hidden=hidden_dims, n_classes=10)
    loss_module = CrossEntropyModule()
    
    # TODO: Training loop including validation
    def early_stopper_wrapper(min_delta, patience, mode):
        best_weights = None
        best_epoch = 0
        epochs_since_improvement = 0

        def check_early_stopping(current_epoch, current_weights, current_metric):
            nonlocal best_weights, best_epoch, epochs_since_improvement
            if mode == 'min':
                if current_metric < best_epoch - min_delta:
                    best_epoch = current_metric
                    epochs_since_improvement = 0
                    best_weights = current_weights
                else:
                    epochs_since_improvement += 1
            elif mode == 'max':
                if current_metric > best_epoch + min_delta:
                    best_epoch = current_metric
                    epochs_since_improvement = 0
                    best_weights = current_weights
                else:
                    epochs_since_improvement += 1

            if epochs_since_improvement >= patience:
                return True, best_weights
            return False, None

        return check_early_stopping
    early_stopper = early_stopper_wrapper(min_delta=0.001, patience=5, mode='max')
    val_accuracies = []
    train_losses = []
    train_accuracies = []
    best_val_accuracy = 0.0
    best_model = None
    
    for epoch in range(epochs):
        # Training phase
        epoch_losses = []
        epoch_train_accuracies = []
        
        for images, labels in tqdm(cifar10_loader['train'], desc=f'Epoch {epoch+1}/{epochs}'):
            # Flatten images from (batch_size, C, H, W) to (batch_size, C*H*W)
            images_flat = images.reshape(images.shape[0], -1)
            
            # Forward pass
            predictions = model.forward(images_flat)
            loss = loss_module.forward(predictions, labels)
            
            # Backward pass
            dout = loss_module.backward(predictions, labels)
            model.backward(dout)
            
            # Update parameters using SGD
            for layer in model.layers:
                if hasattr(layer, 'params') and layer.params['weight'] is not None:
                    layer.params['weight'] -= lr * layer.grads['weight']
                    layer.params['bias'] -= lr * layer.grads['bias']
            
            # Track metrics
            epoch_losses.append(loss)
            batch_accuracy = accuracy(predictions, labels)
            epoch_train_accuracies.append(batch_accuracy)
        
        # Calculate average training metrics for the epoch
        avg_train_loss = np.mean(epoch_losses)
        avg_train_accuracy = np.mean(epoch_train_accuracies)
        train_losses.append(avg_train_loss)
        train_accuracies.append(avg_train_accuracy)
        
        # Validation phase
        val_accuracy = evaluate_model(model, cifar10_loader['validation'])
        val_accuracies.append(val_accuracy)
        
        print(f'Epoch {epoch+1}/{epochs} - Train Loss: {avg_train_loss:.4f}, Train Acc: {avg_train_accuracy:.4f}, Val Acc: {val_accuracy:.4f}')
        
        # Save best model
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            best_model = deepcopy(model)
          
        early_stop = early_stopper(epoch, model, val_accuracy)
        if early_stop[0]:
            print(f'Early stopping at epoch {epoch+1}')
            if early_stop[1] is not None:
                model = early_stop[1]
            break
    
    # TODO: Test best model
    test_accuracy = evaluate_model(best_model, cifar10_loader['test'])
    print(f'\nTest Accuracy: {test_accuracy:.4f}')
    
    # TODO: Add any information you might want to save for plotting
    logging_dict = {
        'train_losses': train_losses,
        'train_accuracies': train_accuracies,
        'val_accuracies': val_accuracies,
        'test_accuracy': test_accuracy,
        'best_val_accuracy': best_val_accuracy
    }
    
    # Return the best model
    model = best_model
    #######################
    # END OF YOUR CODE    #
    #######################

    return model, val_accuracies, test_accuracy, logging_dict


if __name__ == '__main__':
    # Command line arguments
    parser = argparse.ArgumentParser()
    
    # Model hyperparameters
    parser.add_argument('--hidden_dims', default=[512, 256], type=int, nargs='+',
                        help='Hidden dimensionalities to use inside the network. To specify multiple, use " " to separate them. Example: "256 128"')
    
    # Optimizer hyperparameters
    parser.add_argument('--lr', default=0.001, type=float,
                        help='Learning rate to use')
    parser.add_argument('--batch_size', default=128, type=int,
                        help='Minibatch size')

    # Other hyperparameters
    parser.add_argument('--epochs', default=100, type=int,
                        help='Max number of epochs')
    parser.add_argument('--seed', default=42, type=int,
                        help='Seed to use for reproducing results')
    parser.add_argument('--data_dir', default='data/', type=str,
                        help='Data directory where to store/find the CIFAR10 dataset.')

    args = parser.parse_args()
    kwargs = vars(args)

    train(**kwargs)
    # Feel free to add any additional functions, such as plotting of the loss curve here
    