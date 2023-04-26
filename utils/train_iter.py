import wandb

def train(args, model, optimizer, scheduler, train_iter, val_iter, logger=None):

    for i in range(args.epoch):

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
            print("Epoch: {}, Iter: {}, Loss: {}".format(i, iter_idx, loss.item()))
            #wandb.log({'loss': loss.item()})
            if iter_idx % args.val_freq == 0:
                validate(args, model, val_iter, logger)
            pass
        scheduler.step()
        # save model
        # log
        pass
    pass

def validate(args, model, val_iter, logger=None):
    for iter_idx in range(args.max_val_iters):
        batch = next(val_iter)
    pass