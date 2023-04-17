import os
import argparse

from torch.utils.data import DataLoader
from dataset.linemod import LMODataset


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Training with CorDi')
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument('--data_folder', type=str, default='./data/')
    parser.add_argument('--dataset', type=str, default='lm')
    parser.add_argument('--data_augmentation', type=bool, default=True)
    parser.add_argument('--rotated', type=bool, default=False)
    parser.add_argument('--rot_factor', type=float, default=1.)
    parser.add_argument('--points_limit', type=int, default=10000)

    args = parser.parse_args()
    
    train_loader = DataLoader(LMODataset())