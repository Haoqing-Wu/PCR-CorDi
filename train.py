import os
import argparse
import torch
import wandb
from torch.utils.data import DataLoader
from dataset.linemod import LMODataset
from utils.common import *
from utils.train_iter import *
from models.cordi import Cordi




if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Training with CorDi')
    # Dataset
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument('--train_batch_size', type=int, default=128)
    parser.add_argument('--val_batch_size', type=int, default=128)
    parser.add_argument('--data_folder', type=str, default='./data/')
    parser.add_argument('--dataset', type=str, default='lm')
    parser.add_argument('--data_from_pkl', type=bool, default=False)
    parser.add_argument('--gt_vis', type=bool, default=False)
    parser.add_argument('--data_augmentation', type=bool, default=True)
    parser.add_argument('--augment_noise', type=float, default=0.0001)
    parser.add_argument('--rotated', type=bool, default=False)
    parser.add_argument('--rot_factor', type=float, default=1.)
    parser.add_argument('--points_limit', type=int, default=50)
    # Model
    parser.add_argument('--latent_dim', type=int, default=256)
    parser.add_argument('--num_steps', type=int, default=100)
    parser.add_argument('--beta_1', type=float, default=1e-4)
    parser.add_argument('--beta_T', type=float, default=0.02)
    parser.add_argument('--sched_mode', type=str, default='linear')
    parser.add_argument('--residual', type=eval, default=True, choices=[True, False])
    parser.add_argument('--flexibility', type=float, default=0.0)
    # Optimizer and scheduler
    parser.add_argument('--lr', type=float, default=0.002)
    parser.add_argument('--weight_decay', type=float, default=0.0)
    parser.add_argument('--end_lr', type=float, default=1e-4)
    parser.add_argument('--sched_start_epoch', type=int, default=100)
    parser.add_argument('--sched_end_epoch', type=int, default=300)
    # Training
    parser.add_argument('--logging', type=bool, default=True)
    parser.add_argument('--epoch', type=int, default=1000)
    parser.add_argument('--save_freq', type=int, default=10)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--max_train_iters', type=int, default=100)
    parser.add_argument('--max_val_iters', type=int, default=1)
    parser.add_argument('--start_val_epoch', type=int, default=10)
    parser.add_argument('--val_freq', type=int, default=10)


    args = parser.parse_args()
    
    # set logger
    if args.logging:
        wandb.init(project='cordi', config=args)

    # set dataset
    train_data = LMODataset(args, mode='train')
    test_data = LMODataset(args, mode='test')
    train_iter = get_iterator(DataLoader(train_data,
                                         batch_size=args.train_batch_size, 
                                         shuffle=True, 
                                         num_workers=args.workers,
                                         drop_last=True))
    val_iter = get_iterator(DataLoader(test_data,
                                        batch_size=args.val_batch_size,
                                        shuffle=False,
                                        num_workers=args.workers,
                                        drop_last=True))
    # set model
    model = Cordi(args).to(args.device)
    # set optimizer
    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=args.lr,
                                 weight_decay=args.weight_decay)
    # set scheduler
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.8)
    scheduler = get_linear_scheduler(optimizer,
                                     start_epoch=args.sched_start_epoch,
                                     end_epoch=args.sched_end_epoch,
                                     start_lr=args.lr,
                                     end_lr=args.end_lr
)
    # Training
    train(args, model, optimizer, scheduler, train_iter, val_iter)

