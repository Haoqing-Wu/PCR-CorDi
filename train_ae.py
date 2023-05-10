import os
import argparse
import torch
import wandb
from torch.utils.data import DataLoader
from dataset.linemod import LMODataset
from utils.common import *
from utils.train_iter import *
from models.cordi import Cordi
from models.autoencoder import Autoencoder




if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Pretraining feature extractor autoencoder')
    # Dataset
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument('--train_batch_size', type=int, default=32)
    parser.add_argument('--val_batch_size', type=int, default=8)
    parser.add_argument('--data_folder', type=str, default='./data/')
    parser.add_argument('--ckpt_folder', type=str, default='./ckpt/')
    parser.add_argument('--dataset', type=str, default='lm')
    parser.add_argument('--data_from_pkl', type=bool, default=False)
    parser.add_argument('--gt_vis', type=bool, default=False)
    parser.add_argument('--data_augmentation', type=bool, default=True)
    parser.add_argument('--augment_noise', type=float, default=0.0001)
    parser.add_argument('--rotated', type=bool, default=False)
    parser.add_argument('--rot_factor', type=float, default=1.)
    parser.add_argument('--points_limit', type=int, default=500)
    # Model
    parser.add_argument('--resume', type=bool, default=False)
    parser.add_argument('--latent_dim', type=int, default=256)
    parser.add_argument('--num_steps', type=int, default=100)
    parser.add_argument('--beta_1', type=float, default=1e-4)
    parser.add_argument('--beta_T', type=float, default=0.02)
    parser.add_argument('--sched_mode', type=str, default='linear')
    parser.add_argument('--residual', type=eval, default=True, choices=[True, False])
    parser.add_argument('--flexibility', type=float, default=0.0)

    # Optimizer and scheduler
    parser.add_argument('--lr', type=float, default=0.0005)
    parser.add_argument('--weight_decay', type=float, default=0.0)
    # Training
    parser.add_argument('--logging', type=bool, default=True)
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
    tr_loader = DataLoader(LMODataset(args, mode='train'),
                           batch_size=args.train_batch_size, 
                           shuffle=True, 
                           num_workers=args.workers,
                           drop_last=True)
    val_loader = DataLoader(LMODataset(args, mode='test'),
                            batch_size=args.val_batch_size,
                            shuffle=False,
                            num_workers=args.workers,
                            drop_last=True)
    # set model
    model = Autoencoder(args).to(args.device)
    if args.resume:
        model.load_state_dict(torch.load(os.path.join(args.ckpt_folder, 'autoencoder.pt')))
    # set optimizer
    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=args.lr,
                                 weight_decay=args.weight_decay)
    # set scheduler
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.98)
    # Training
    
    for epoch in range(args.epoch):
        for idx, batch in enumerate(tr_loader):
            # set data                 
            tgt = batch['tgt_pcd'].to(args.device)
            # set optimizer
            optimizer.zero_grad()
            model.train()
            # forward
            loss, _= model.get_loss(tgt)
            # backward
            loss.backward()
            # update
            optimizer.step()
            # log
            lr = optimizer.param_groups[0]['lr']
            print("[train]Epoch: {0:3d}, Iter: {1:3d}, Loss: {2:6.4f}, lr: {3:7.6f}".format(
                epoch, idx, loss.item(), lr))
            if args.logging:
                wandb.log({'loss': loss.item(), 'lr': lr})
            
        if epoch % args.val_freq == 0 and epoch >= args.start_val_epoch:
            with torch.no_grad():
                for idx, batch in enumerate(val_loader):
                    # set data                 
                    tgt = batch['tgt_pcd'].to(args.device)
                    # set optimizer
                    optimizer.zero_grad()
                    model.eval()
                    # forward
                    loss, recon = model.get_loss(tgt)
                    # log
                    print("[val]Epoch: {0:3d}, Iter: {1:3d}, Loss: {2:6.4f}".format(
                        epoch, idx, loss.item()))
                    if args.logging:
                        wandb.log({'val_loss': loss.item()})
                        ref = wandb.Object3D(tgt[0].cpu().numpy(), caption='ref')
                        pred = wandb.Object3D(recon[0].cpu().numpy(), caption='pred')
                        wandb.log({'ref': ref, 'pred': pred})

                    break
        # save the model
        if epoch % 20 == 0:
            torch.save(model.state_dict(), os.path.join(args.ckpt_folder, 'autoencoder.pt'))
        scheduler.step()
        # save model
        # log
        pass
    pass

