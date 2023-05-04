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
            corr_vector = batch['corr_vector'].to(args.device)
            src = batch['src_pcd'].to(args.device)
            tgt = batch['tgt_pcd'].to(args.device)
            # set optimizer
            optimizer.zero_grad()
            model.train()
            # forward
            loss = model.get_loss(corr_vector, src, tgt)
            
            # backward
            loss.backward()
            # update
            optimizer.step()
            # log
            lr = optimizer.param_groups[0]['lr']
            print("[train]Epoch: {}, Iter: {}, Loss: {}, lr: {}".format(epoch, iter_idx, loss.item(), lr))
            if args.logging:
                wandb.log({'loss': loss.item()})
            
        if epoch % args.val_freq == 0 and epoch >= args.start_val_epoch:
            validate(args, model, train_iter, logger)  
            pass
        scheduler.step()
        # save model
        # log
        pass
    pass

def validate(args, model, val_iter, logger=None):
    for iter_idx in range(args.max_val_iters):
        batch = next(val_iter)
        src = batch['src_pcd'].to(args.device)
        tgt = batch['tgt_pcd'].to(args.device)
        corr_vector = batch['corr_vector']
        rot = batch['rot']
        trans = batch['trans']
        shift = batch['shift']
        scale = batch['scale']

        corr_vector_T = torch.randn([args.val_batch_size, tgt.shape[1], 3]).to(args.device)
        with torch.no_grad():
            model.eval()
            z = torch.randn([args.val_batch_size, args.latent_dim]).to(args.device)
            samples = model.sample(z, src, tgt, args.flexibility)
            for i in range(args.val_batch_size):
                gt_corr_vector = corr_vector[i].numpy()
                gt_rot = rot[i].numpy()
                gt_trans = trans[i].numpy()
                src_pcd = src[i].cpu().numpy()
                tgt_pcd = tgt[i].cpu().numpy()
                shift_v = shift[i].numpy()
                scale_v = scale[i].numpy()
                pred_vector = samples[i].cpu().numpy() * scale_v + shift_v	
                pred_src_pcd = tgt_pcd + pred_vector
                pred_corr, _ = get_corr_k(pred_src_pcd, src_pcd)
                gt_corr, _ = get_corr_k(tgt_pcd, src_pcd, transform=True, rot=gt_rot, trans=gt_trans)
                outlier_r = get_outlier_ratio(pred_corr, gt_corr)

            
                if iter_idx == 0 and i == 0:
                    #corr_visualisation(src_pcd, tgt_pcd, pred_corr_matrix, gt_corr, gt_rot, gt_trans)
                    gen_visualisation(src_pcd, pred_src_pcd, tgt_pcd)
                    pass
                print("[Val]Iter: {0}, outlier_ratio: {1}".format(iter_idx, outlier_r))
                if args.logging:
                    wandb.log({'outlier_ratio': outlier_r})

    pass