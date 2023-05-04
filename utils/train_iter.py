import wandb
import torch
from dataset.bop_utils import *
from random import shuffle

def train(args, model, optimizer, scheduler, tr_loader, val_loader, logger=None):

    for epoch in range(args.epoch):

        # train
        for idx, batch in enumerate(tr_loader):
            # set data        
            corr = batch['corr_matrix'].to(args.device)           
            src = batch['src_pcd'].to(args.device)
            tgt = batch['tgt_pcd'].to(args.device)
            # set optimizer
            optimizer.zero_grad()
            model.train()
            # forward
            loss, loss_o, loss_b = model.get_loss(corr, src, tgt)
            
            # backward
            loss.backward()
            # update
            optimizer.step()
            # log
            lr = optimizer.param_groups[0]['lr']
            print("[train]Epoch: {0:3d}, Iter: {1:3d}, Loss: {2:6.4f}, loss_o: {3:6.4f}, loss_b: {4:6.4f}, lr: {5:7.6f}".format(
                epoch, idx, loss.item(), loss_o.item(), loss_b.item(), lr))
            if args.logging:
                wandb.log({'loss': loss.item(), 'loss_o': loss_o.item(), 'loss_b': loss_b.item(), 'lr': lr})
            
        if epoch % args.val_freq == 0 and epoch >= args.start_val_epoch:
            validate(args, model, tr_loader, logger)  
        scheduler.step()
        # save model
        # log
        pass
    pass

def validate(args, model, val_loader, logger=None):

    for idx, batch in enumerate(val_loader):
        src = batch['src_pcd'].to(args.device)
        tgt = batch['tgt_pcd'].to(args.device)
        corr = batch['corr_matrix']
        rot = batch['rot']
        trans = batch['trans']

        corr_T = torch.randn([args.val_batch_size, tgt.shape[1], src.shape[1]]).to(args.device)
        x_T = model.get_x_t(corr.to(args.device), args.num_steps)
        with torch.no_grad():
            model.eval()
            samples = model.sample(corr_T, src, tgt, args.flexibility)
            for i in range(args.val_batch_size):
                gt_corr = corr[i].numpy()
                gt_rot = rot[i].numpy()
                gt_trans = trans[i].numpy()
                src_pcd = src[i].cpu().numpy()
                tgt_pcd = tgt[i].cpu().numpy()
                pred_corr = samples[i].cpu().numpy()
                #f_loss = focal_loss(gt_corr, pred_corr)
                pred_corr_pair = get_corr_from_matrix_topk(samples[i].cpu(), 40)
                pred_corr_matrix = get_corr_matrix(pred_corr_pair, tgt.shape[1], src.shape[1])
                #p_loss = focal_loss(gt_corr, pred_corr_matrix)

                inlier_ratio = corr_visualisation(src_pcd, tgt_pcd, pred_corr_matrix, gt_corr, gt_rot, gt_trans)
                print("[Val]Iter: {0:3d}, inlier_ratio: {1:5.4f}".format(idx, inlier_ratio))
                if args.logging:
                    wandb.log({'inlier_ratio': inlier_ratio})
                break #
        break # 

    pass