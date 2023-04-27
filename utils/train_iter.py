import wandb
import torch
from dataset.bop_utils import *
from random import shuffle

def train(args, model, optimizer, scheduler, train_iter, val_iter, logger=None):

    for epoch in range(args.epoch):

        # train
        for iter_idx in range(args.max_train_iters):
            # set data
            batch = next(train_iter)
            corr = batch['corr_matrix'].to(args.device)
            src = batch['src_pcd'].to(args.device)
            tgt = batch['tgt_pcd'].to(args.device)
            # set optimizer
            optimizer.zero_grad()
            model.train()
            # forward
            loss = model.get_loss(corr, src, tgt)
            
            # backward
            loss.backward()
            # update
            optimizer.step()
            # log
            lr = optimizer.param_groups[0]['lr']
            print("[train]Epoch: {}, Iter: {}, Loss: {}".format(epoch, iter_idx, loss.item()))
            wandb.log({'loss': loss.item()})
            
        if epoch % args.val_freq == 0 and epoch >= args.start_val_epoch:
            validate(args, model, val_iter, logger)  
        scheduler.step()
        # save model
        # log
        pass
    pass

def validate(args, model, val_iter, logger=None):
    for iter_idx in range(args.max_val_iters):
        batch = next(val_iter)
        corr = batch['corr_matrix'].to(args.device)
        src = batch['src_pcd'].to(args.device)
        tgt = batch['tgt_pcd'].to(args.device)
        corr_T = torch.randn([args.val_batch_size, tgt.shape[1], src.shape[1]]).to(args.device)
        with torch.no_grad():
            samples = model.sample(corr_T, src, tgt, args.flexibility)
            for i in range(args.val_batch_size):
                gt_corr = corr[i].cpu().numpy()
                pred_corr = samples[i].cpu().numpy()
                f_loss = focal_loss(gt_corr, pred_corr)
                pred_corr_pair = get_corr_from_matrix_topk(corr[i].cpu(),400)
                pred_corr_matrix = get_corr_matrix(pred_corr_pair, tgt.shape[1], src.shape[1])
                p_loss = focal_loss(gt_corr, pred_corr_matrix)
                print("[Val]Iter: {0}, p_loss: {1}, c_loss: {2}".format(iter_idx, f_loss, p_loss))
                wandb.log({'p_loss': p_loss, 'c_loss': f_loss})


    pass