
def train(args, model, optimizer, scheduler, data_iter, logger=None):

    for i in range(args.epoch):

        # train
        for iter_idx in range(args.max_iters):
            # set data
            batch = next(data_iter)
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
            pass
        scheduler.step()
        # save model
        # log
        pass
    pass