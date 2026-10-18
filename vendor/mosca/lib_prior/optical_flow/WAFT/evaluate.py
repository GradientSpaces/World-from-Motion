import sys
import math
import argparse
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data

from config.parser import parse_args

from tqdm import tqdm
from model import fetch_model
from utils.utils import resize_data, load_ckpt

from dataloader.flow.chairs import FlyingChairs
from dataloader.flow.things import FlyingThings3D
from dataloader.flow.sintel import MpiSintel
from dataloader.flow.kitti import KITTI
from dataloader.flow.spring import Spring
from dataloader.flow.hd1k import HD1K
from dataloader.stereo.tartanair import TartanAir

from inference_tools import InferenceWrapper, AverageMeter

val_loss = AverageMeter()
val_epe = AverageMeter()
val_fl = AverageMeter()
val_px1 = AverageMeter()

def reset_all_metrics():
    val_loss.reset()
    val_epe.reset()
    val_fl.reset()
    val_px1.reset()

def update_metrics(args, output, flow_gt, valid):
    flow = output['flow'][-1]
    batch_size = flow.shape[0]
    epe = torch.sum((flow - flow_gt)**2, dim=1).sqrt()
    mag = torch.sum(flow_gt**2, dim=1).sqrt()
    val = valid >= 0.5
    out = ((epe > 3.0) & ((epe/mag) > 0.05)).float()
    px1 = (epe < 1.0).float()
    nf = []
    for i in range(len(output['flow'])):                 
        raw_b = output['info'][i][:, 2:]
        log_b = torch.zeros_like(raw_b)
        weight = output['info'][i][:, :2]
        log_b[:, 0] = torch.clamp(raw_b[:, 0], min=0, max=args.var_max)
        log_b[:, 1] = torch.clamp(raw_b[:, 1], min=args.var_min, max=0)
        term2 = ((flow_gt - output['flow'][i]).abs().unsqueeze(2)) * (torch.exp(-log_b).unsqueeze(1))
        term1 = weight - math.log(2) - log_b
        nf_loss = torch.logsumexp(weight, dim=1, keepdim=True) - torch.logsumexp(term1.unsqueeze(1) - term2, dim=2)
        nf.append(nf_loss.mean(dim=1))

    loss = torch.zeros_like(nf[-1])
    for i in range(len(nf)):
        loss += (args.gamma ** (len(nf) - i - 1)) * nf[i]
    for i in range(batch_size):
        val_epe.update(epe[i][val[i]].mean().item(), 1)
        val_px1.update(px1[i][val[i]].mean().item(), 1)
        val_fl.update(100 * out[i][val[i]].sum().item(), val[i].sum().item())
        val_loss.update(loss[i][val[i]].mean().item(), 1)

@torch.no_grad()
def validate_sintel(args, model):
    """ Peform validation using the Sintel (train) split """
    for dstype in ['clean', 'final']:
        reset_all_metrics()
        val_dataset = MpiSintel(split='training', dstype=dstype)
        val_loader = data.DataLoader(val_dataset, batch_size=1, 
            pin_memory=False, shuffle=False, num_workers=16, drop_last=False)
        pbar = tqdm(total=len(val_loader))
        print(f"load data success {len(val_loader)}")
        for i_batch, data_blob in enumerate(val_loader):
            image1, image2, flow_gt, valid = [x.cuda(non_blocking=True) for x in data_blob]
            output = model.calc_flow(image1, image2)
            update_metrics(args, output, flow_gt, valid)
            pbar.update(1)
        pbar.close()
        print(f"Validation {dstype} EPE: {val_epe.avg}, 1px: {100 * (1 - val_px1.avg)}")

@torch.no_grad()
def validate_kitti(args, model):
    """ Peform validation using the KITTI-2015 (train) split """
    val_dataset = KITTI(split='training')
    val_loader = data.DataLoader(val_dataset, batch_size=1, 
        pin_memory=False, shuffle=False, num_workers=16, drop_last=False)
    print(f"load data success {len(val_loader)}")
    reset_all_metrics()
    for i_batch, data_blob in enumerate(val_loader):
        image1, image2, flow_gt, valid_gt = [x.cuda(non_blocking=True) for x in data_blob]
        output = model.calc_flow(image1, image2)
        update_metrics(args, output, flow_gt, valid_gt)
    
    print("Validation KITTI: %f, %f" % (val_epe.avg, val_fl.avg))

@torch.no_grad()
def validate_spring(args, model):
    """ Peform validation using the Spring (val) split """
    val_dataset = Spring(split='val') #+ Spring(split='train')
    val_loader = data.DataLoader(val_dataset, batch_size=1, 
        pin_memory=False, shuffle=False, num_workers=16, drop_last=False)
    
    reset_all_metrics()
    print(f"load data success {len(val_loader)}")
    pbar = tqdm(total=len(val_loader))
    for i_batch, data_blob in enumerate(val_loader):
        image1, image2, flow_gt, valid = [x.cuda(non_blocking=True) for x in data_blob]
        output = model.calc_flow(image1, image2)
        update_metrics(args, output, flow_gt, valid)
        pbar.update(1)

    pbar.close()
    print(f"Validation Spring EPE: {val_epe.avg}, 1px: {100 * (1 - val_px1.avg)}, loss: {val_loss.avg}")

def eval(args):
    args.gpus = [0]
    model = fetch_model(args)
    load_ckpt(model, args.ckpt)
    model = model.cuda()
    model.eval()
    wrapped_model = InferenceWrapper(model, scale=args.scale, train_size=args.image_size, pad_to_train_size=False, tiling=False)

    with torch.no_grad():
        if args.dataset == 'spring':
            validate_spring(args, wrapped_model)
        elif args.dataset == 'sintel':
            validate_sintel(args, wrapped_model)
        elif args.dataset == 'kitti':
            validate_kitti(args, wrapped_model)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', help='experiment configure file name', required=True, type=str)
    parser.add_argument('--ckpt', help='checkpoint path', required=True, type=str)
    parser.add_argument('--dataset', help='dataset to evaluate on', choices=['sintel', 'kitti', 'spring'], required=True, type=str)
    parser.add_argument('--scale', help='scale factor for input images', default=0.0, type=float)
    args = parse_args(parser)
    eval(args)

if __name__ == '__main__':
    main()
