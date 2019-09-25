import torch
import torch.nn as nn


def cycle_rgb_weights(weights, n):
    """Repeat RGB weights n times. Assumes channels are dim 1"""
    slices = [(c % 3, c % 3 + 1) for c in range(n)]  # slice a:a+1 to keep dims
    new_weights = torch.cat([
        weights[:, a:b, :, :] for a, b in slices
    ], dim=1)
    return new_weights


def avg_rgb_weights(weights):
    new_weights = weights.mean(dim=1, keepdim=True)
    return new_weights


def transfer_weights(pretrained, replacement, method='cycle'):
    """
    Transform pretrained weights to be used for a layer with a different number of channels.
    """
    if method == 'cycle':
        n = replacement.in_channels
        weights = cycle_rgb_weights(pretrained.weight, n)
    elif method == 'avg':
        weights = avg_rgb_weights(pretrained.weight)
    else:
        raise NotImplementedError('`method` must be "cycle" or "avg", received {}'.format(method))
    replacement.weight = nn.Parameter(weights)
    return replacement
