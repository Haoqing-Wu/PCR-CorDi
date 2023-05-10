import torch
import numpy as np
from chamfer_distance import ChamferDistance

def get_iterator(iterable):

    iterator = iterable.__iter__()
    while True:
        try:
            yield iterator.__next__()
        except StopIteration:
            iterator = iterable.__iter__()

def reparameterize_gaussian(mean, logvar):
    std = torch.exp(0.5 * logvar)
    eps = torch.randn(std.size()).to(mean)
    return mean + std * eps

def get_weight_tensor_from_corr(corr, weight_a, weight_b):
    weight_tensor = torch.where(corr == 1, torch.tensor(weight_a), torch.tensor(weight_b))
    return weight_tensor

def weighted_mse_loss(tensor_a, tensor_b, weight_tensor):
    squared_diff = torch.pow(tensor_a - tensor_b, 2)
    weighted_squared_diff = squared_diff * weight_tensor
    # get all none zero elements
    weighted_squared_diff = weighted_squared_diff[weighted_squared_diff != 0]
    mean_weighted_squared_diff = torch.mean(weighted_squared_diff)
    return mean_weighted_squared_diff

def chamfer(pcd1, pcd2):
    chamfer_dist = ChamferDistance()
    dist1, dist2, _, _ = chamfer_dist(pcd1, pcd2)
    loss = torch.mean(dist1) + torch.mean(dist2)
    return loss
