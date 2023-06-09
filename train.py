import os
import argparse
import torch
import wandb
import time
from torch.utils.data import DataLoader
from dataset.linemod import LMODataset
from utils.common import *
from utils.train_iter import *
from models.cordi import Cordi
from dataset.dataloader import train_dataloader




if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Training with CorDi')
    # Dataset
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument('--train_batch_size', type=int, default=1)
    parser.add_argument('--val_batch_size', type=int, default=1)
    parser.add_argument('--data_folder', type=str, default='./data/')
    parser.add_argument('--ckpt_folder', type=str, default='./ckpt/')
    parser.add_argument('--dataset', type=str, default='lm')
    parser.add_argument('--reload_data', type=bool, default=False)
    parser.add_argument('--overfit', type=int, default=1)
    parser.add_argument('--data_augmentation', type=bool, default=True)
    parser.add_argument('--augment_noise', type=float, default=0.0005)
    parser.add_argument('--rotated', type=bool, default=False)
    parser.add_argument('--rot_factor', type=float, default=1.)
    parser.add_argument('--points_limit', type=int, default=1000)
    # Model
    parser.add_argument('--pretrained', type=bool, default=True)
    parser.add_argument('--latent_dim', type=int, default=256)
    parser.add_argument('--num_steps', type=int, default=100)
    parser.add_argument('--beta_1', type=float, default=1e-4)
    parser.add_argument('--beta_T', type=float, default=0.02)
    parser.add_argument('--sched_mode', type=str, default='linear')
    parser.add_argument('--residual', type=eval, default=True, choices=[True, False])
    parser.add_argument('--flexibility', type=float, default=0.0)

    parser.add_argument('--KPConv_num_stages', type=int, default=3)
    parser.add_argument('--KPConv_init_voxel_size', type=float, default=0.003)
    parser.add_argument('--KPConv_init_radius', type=float, default=0.01)

    # Optimizer and scheduler
    parser.add_argument('--lr', type=float, default=0.0005)
    parser.add_argument('--weight_decay', type=float, default=0.0)
    # Training
    parser.add_argument('--logging', type=bool, default=False)
    parser.add_argument('--epoch', type=int, default=2000)
    #parser.add_argument('--save_freq', type=int, default=10)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--max_train_iters', type=int, default=100)
    parser.add_argument('--max_val_iters', type=int, default=1)
    parser.add_argument('--start_val_epoch', type=int, default=5)
    parser.add_argument('--val_freq', type=int, default=5)


    args = parser.parse_args()
    
    # set logger
    if args.logging:
        wandb.init(project='cordi', config=args)

    # set dataset
    start_time = time.time()
    tr_loader, val_loader, neighbor_limits = train_dataloader(args)
    loading_time = time.time() - start_time
    print('Data loader created: {:.3f}s collapsed.'.format(loading_time))
    print('Calibrate neighbors: {}.'.format(neighbor_limits))
    
    # set model
    model = Cordi(args).to(args.device)
    # set optimizer
    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=args.lr,
                                 weight_decay=args.weight_decay)
    # set scheduler
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.98)
    # Training
    train(args, model, optimizer, scheduler, tr_loader, val_loader)

