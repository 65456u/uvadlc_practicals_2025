################################################################################
# MIT License
#
# Copyright (c) 2022
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to conditions.
#
# Author: Deep Learning Course | Autumn 2022
# Date Created: 2022-11-25
################################################################################

import torch
from torchvision.utils import make_grid
import numpy as np


def sample_reparameterize(mean, std):
    """
    Perform the reparameterization trick to sample from a distribution with the given mean and std
    Inputs:
        mean - Tensor of arbitrary shape and range, denoting the mean of the distributions
        std - Tensor of arbitrary shape with strictly positive values. Denotes the standard deviation
              of the distribution
    Outputs:
        z - A sample of the distributions, with gradient support for both mean and std.
            The tensor should have the same shape as the mean and std input tensors.
    """
    assert not (std < 0).any().item(), "The reparameterization trick got a negative std as input. " + \
                                       "Are you sure your input is std and not log_std?"
    #######################
    # PUT YOUR CODE HERE  #
    #######################
    eps = torch.randn_like(std)
    z = mean + std * eps
    #######################
    # END OF YOUR CODE    #
    #######################
    return z


def KLD(mean, log_std):
    """
    Calculates the Kullback-Leibler divergence of given distributions to unit Gaussians over the last dimension.
    See the definition of the regularization loss in Section 1.4 for the formula.
    Inputs:
        mean - Tensor of arbitrary shape and range, denoting the mean of the distributions.
        log_std - Tensor of arbitrary shape and range, denoting the log standard deviation of the distributions.
    Outputs:
        KLD - Tensor with one less dimension than mean and log_std (summed over last dimension).
              The values represent the Kullback-Leibler divergence to unit Gaussians.
    """

    #######################
    # PUT YOUR CODE HERE  #
    #######################
    var = torch.exp(2 * log_std)
    KLD = 0.5 * (var + mean**2 - 1 - 2 * log_std).sum(dim=-1)
    #######################
    # END OF YOUR CODE    #
    #######################
    return KLD


def elbo_to_bpd(elbo, img_shape):
    """
    Converts the summed negative log likelihood given by the ELBO into the bits per dimension score.
    Inputs:
        elbo - Tensor of shape [batch_size]
        img_shape - Shape of the input images, representing [batch, channels, height, width]
    Outputs:
        bpd - The negative log likelihood in bits per dimension for the given image.
    """
    #######################
    # PUT YOUR CODE HERE  #
    #######################
    num_dims = torch.tensor(img_shape[1:]).prod()
    bpd = elbo * torch.log2(torch.tensor(torch.e)) / num_dims
    #######################
    # END OF YOUR CODE    #
    #######################
    return bpd


@torch.no_grad()
def visualize_manifold(decoder, grid_size=20):
    """
    Visualize a manifold over a 2 dimensional latent space. The images in the manifold
    should represent the decoder's output means (not binarized samples of those).
    Inputs:
        decoder - Decoder model such as LinearDecoder or ConvolutionalDecoder.
        grid_size - Number of steps/images to have per axis in the manifold.
                    Overall you need to generate grid_size**2 images, and the distance
                    between different latents in percentiles is 1/grid_size
    Outputs:
        img_grid - Grid of images representing the manifold.
    """

    ## Hints:
    # - You can use the icdf method of the torch normal distribution  to obtain z values at percentiles.
    # - Use the range [0.5/grid_size, 1.5/grid_size, ..., (grid_size-0.5)/grid_size] for the percentiles.
    # - torch.meshgrid might be helpful for creating the grid of values
    # - You can use torchvision's function "make_grid" to combine the grid_size**2 images into a grid
    # - Remember to apply a softmax after the decoder

    #######################
    # PUT YOUR CODE HERE  #
    #######################
    device = next(decoder.parameters()).device

    # Percentiles: [0.5/grid, 1.5/grid, ..., (grid-0.5)/grid]
    p = (torch.arange(grid_size, device=device, dtype=torch.float32) + 0.5) / grid_size

    # Map percentiles to z values using inverse CDF of N(0,1)
    normal = torch.distributions.Normal(
        loc=torch.tensor(0.0, device=device),
        scale=torch.tensor(1.0, device=device),
    )
    z_vals = normal.icdf(p)  # [grid_size]

    # Create grid in latent space
    z1, z2 = torch.meshgrid(z_vals, z_vals, indexing="ij")  # each [grid_size, grid_size]
    z = torch.stack([z1, z2], dim=-1).reshape(-1, 2)        # [grid_size^2, 2]

    # Decode
    logits = decoder(z)

    # Ensure logits are [B, K, H, W] for softmax
    if logits.dim() == 4:
        # common: [B, K, H, W] or [B, H, W, K]
        if logits.size(1) < logits.size(-1) and logits.size(1) < logits.size(-2):
            # assume [B, K, H, W]
            logits_bkhw = logits
        else:
            # assume [B, H, W, K] -> [B, K, H, W]
            logits_bkhw = logits.permute(0, 3, 1, 2).contiguous()
    elif logits.dim() == 2:
        # If decoder outputs flattened pixels, treat as Bernoulli mean via sigmoid
        # shape [B, M] -> reshape to [B, 1, H, W] is ambiguous, so we error out
        raise ValueError("Decoder output is 2D; expected categorical logits with shape [B,K,H,W] or [B,H,W,K].")
    else:
        raise ValueError(f"Unexpected decoder output shape: {tuple(logits.shape)}")

    # Convert logits -> probabilities
    probs = torch.softmax(logits_bkhw, dim=1)  # [B, K, H, W]
    K = probs.size(1)

    # Convert categorical probabilities to an "output mean" image in [0,1]
    # mean = E[value] where value in {0,...,K-1} normalized by (K-1)
    values = torch.linspace(0.0, 1.0, steps=K, device=device).view(1, K, 1, 1)
    mean_img = (probs * values).sum(dim=1, keepdim=True)  # [B, 1, H, W]

    # Make a grid image: grid_size x grid_size
    img_grid = make_grid(mean_img, nrow=grid_size, padding=2)
    #######################
    # END OF YOUR CODE    #
    #######################

    return img_grid

