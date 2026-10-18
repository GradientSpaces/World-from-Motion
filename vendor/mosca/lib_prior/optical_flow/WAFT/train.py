import sys
import os
import argparse
import numpy as np
import torch
import time
import torch.optim as optim

from config.parser import parse_args
from model import fetch_model
from dataloader.loader import fetch_dataloader
from utils.utils import load_ckpt
from utils.ddp_utils import *
from criterion.loss import sequence_loss

import wandb

os.system("export KMP_INIT_AT_FORK=FALSE")

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def fetch_optimizer(args, model):
    """ Create the optimizer and learning rate scheduler """
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wdecay, eps=args.epsilon)
    scheduler = optim.lr_scheduler.OneCycleLR(optimizer, args.lr, args.num_steps + 100,
        pct_start=0.05, cycle_momentum=False, anneal_strategy='linear')

    return optimizer, scheduler

def train(args, rank=0, world_size=1, use_ddp=False):
    """ Full training loop """
    device_id = rank
    model = fetch_model(args).to(device_id)
    if rank == 0:
        avg_loss = AverageMeter()
        avg_epe = AverageMeter()

        if args.algorithm == 'waftv2':
            args.exp_name = f"{args.algorithm}-{args.feature_encoder}-{args.seed}"
        else:
            args.feature_encoder = 'dav2'
            args.exp_name = f"{args.algorithm}-{args.feature_encoder}-{args.seed}"

        wandb.init(
            project=args.name,
            name=args.exp_name,
        )
    if args.restore_ckpt is not None:
        load_ckpt(model, args.restore_ckpt)
        print(f"restore ckpt from {args.restore_ckpt}")

    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[rank], static_graph=True)
    model.train()
    train_loader = fetch_dataloader(args, rank=rank, world_size=world_size, use_ddp=use_ddp)
    optimizer, scheduler = fetch_optimizer(args, model)
    total_steps = 0
    VAL_FREQ = 10000
    epoch = 0
    cnt_overheat = 0
    should_keep_training = True
    while should_keep_training:
        # shuffle sampler
        train_loader.sampler.set_epoch(epoch)
        epoch += 1
        for i_batch, data_blob in enumerate(train_loader):
            optimizer.zero_grad()
            image1, image2, flow, valid = [x.cuda(non_blocking=True) for x in data_blob]
            output = model(image1, image2, flow_gt=flow)
            loss = sequence_loss(output, flow, valid, args.gamma)
            # logger
            if rank == 0:
                if valid.sum() > 0:
                    avg_loss.update(loss.item())
                    epe = (((flow - output['flow'][-1])**2).sum(dim=1)).sqrt()
                    epe = (epe * valid).sum() / valid.sum()
                    avg_epe.update(epe.item())
                    
                if total_steps % 100 == 0:    
                    wandb.log({"loss": avg_loss.avg, "epe": avg_epe.avg})
                    avg_loss.reset()
                    avg_epe.reset()
                    cnt_overheat = 0


            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip) 
            optimizer.step()
            scheduler.step()
            if total_steps % VAL_FREQ == VAL_FREQ - 1 and rank == 0:
                save_dir = os.path.join('checkpoints', str(args.name), str(args.algorithm), str(args.feature_encoder), str(args.seed))
                os.makedirs(save_dir, exist_ok=True)
                torch.save(model.module.state_dict(), os.path.join(save_dir, f"{total_steps+1}.pth"))

            if total_steps > args.num_steps:
                should_keep_training = False
                break
            
            total_steps += 1

    if rank == 0:
        save_dir = os.path.join('checkpoints', str(args.name), str(args.algorithm), str(args.feature_encoder), str(args.seed))
        os.makedirs(save_dir, exist_ok=True)
        torch.save(model.module.state_dict(), os.path.join(save_dir, 'final.pth'))
        wandb.finish()

def main(rank, world_size, args, use_ddp):
    if use_ddp:
        print(f"Using DDP [{rank=} {world_size=}]")
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        random.seed(args.seed)
        setup_ddp(rank, world_size)

    train(args, rank=rank, world_size=world_size, use_ddp=use_ddp)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', help='experiment configure file name', required=True, type=str)
    parser.add_argument('--seed', help='seed', default=42, type=int)
    parser.add_argument('--restore_ckpt', help='restore checkpoint', default=None, type=str)
    args = parse_args(parser)
    smp, world_size = init_ddp()
    if world_size > 1:
        spwn_ctx = mp.spawn(main, nprocs=world_size, args=(world_size, args, True), join=False)
        spwn_ctx.join()
    else:
        main(0, 1, args, False)
    print("Done!")