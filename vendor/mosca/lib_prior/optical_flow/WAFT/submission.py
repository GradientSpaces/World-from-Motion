import os
import cv2
import sys
import argparse
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data

from config.parser import parse_args

from tqdm import tqdm

from model import fetch_model

from utils import frame_utils
from utils.utils import resize_data, load_ckpt
from utils.flow_viz import flow_to_image

from dataloader.flow.sintel import MpiSintel
from dataloader.flow.kitti import KITTI
from dataloader.flow.spring import Spring

from inference_tools import InferenceWrapper, AverageMeter

@torch.no_grad()
def create_spring_submission(args, model, output_path='../spring_submission'):
    """ Create submission for the Sintel leaderboard """
    test_dataset = Spring(split='test', aug_params=None)
    pbar = tqdm(total=len(test_dataset))
    for test_id in range(len(test_dataset)):
        image1, image2, extra_info = test_dataset[test_id]
        frame, scene, cam, direction = extra_info
        image1 = image1[None].cuda()
        image2 = image2[None].cuda()        
        output = model.calc_flow(image1, image2)
        flow = output['flow'][-1]
        flow = flow[0].permute(1, 2, 0).cpu().numpy()
        flow_gt_vis = flow_to_image(flow, convert_to_bgr=True)
        output_dir = os.path.join(output_path, scene, f"flow_{direction}_{cam}")
        output_file = os.path.join(output_dir, f"flow_{direction}_{cam}_{frame:04d}.flo5")
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        cv2.imwrite(os.path.join(output_dir, f"flow_{direction}_{cam}_{frame:04d}.png"), flow_gt_vis)
        frame_utils.writeFlo5File(flow, output_file)
        pbar.update(1)

    pbar.close()

@torch.no_grad()
def create_sintel_submission(args, model, output_path='../sintel_submission'):
    """ Create submission for the Sintel leaderboard """
    for dstype in ['clean', 'final']:
        test_dataset = MpiSintel(split='test', aug_params=None, dstype=dstype)
        flow_prev, sequence_prev = None, None
        pbar = tqdm(total=len(test_dataset))
        for test_id in range(len(test_dataset)):
            image1, image2, (sequence, frame) = test_dataset[test_id]
            image1 = image1[None].cuda()
            image2 = image2[None].cuda()
            output = model.calc_flow(image1, image2)
            flow = output['flow'][-1]
            flow = flow[0].permute(1, 2, 0).cpu().numpy()
            flow_gt_vis = flow_to_image(flow, convert_to_bgr=True)
            output_dir = os.path.join(output_path, dstype, sequence)
            output_file = os.path.join(output_dir, 'frame%04d.flo' % (frame+1))

            if not os.path.exists(output_dir):
                os.makedirs(output_dir)

            frame_utils.writeFlow(output_file, flow)
            cv2.imwrite(os.path.join(output_dir, f"frame{frame+1}.png"), flow_gt_vis)
            sequence_prev = sequence
            pbar.update(1)
        
        pbar.close()

@torch.no_grad()
def create_kitti_submission(args, model, output_path='../kitti_submission'):
    """ Create submission for the Sintel leaderboard """
    test_dataset = KITTI(split='testing', aug_params=None)
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    
    pbar = tqdm(total=len(test_dataset))
    for test_id in range(len(test_dataset)):
        image1, image2, (frame_id, ) = test_dataset[test_id]
        image1 = image1[None].cuda()
        image2 = image2[None].cuda()
        output = model.calc_flow(image1, image2)
        flow = output['flow'][-1]
        flow = flow[0].permute(1, 2, 0).cpu().numpy()
        output_filename = os.path.join(output_path, frame_id)
        flow_gt_vis = flow_to_image(flow, convert_to_bgr=True)
        cv2.imwrite(os.path.join(output_path, f"frame{frame_id}"), flow_gt_vis)
        frame_utils.writeFlowKITTI(output_filename, flow)
        pbar.update(1)

    pbar.close()

def eval(args):
    args.gpus = [0]
    model = fetch_model(args)
    load_ckpt(model, args.ckpt)
    model = model.cuda()
    model.eval()
    wrapped_model = InferenceWrapper(model, scale=args.scale, train_size=args.image_size, pad_to_train_size=False, tiling=False)
    if args.output_path is not None:
        output_path = args.output_path
    else:
        output_path = f'../{args.dataset}_submission'
    with torch.no_grad():
        if args.dataset == 'spring':
            create_spring_submission(args, wrapped_model, output_path=output_path)
        elif args.dataset == 'sintel':
            create_sintel_submission(args, wrapped_model, output_path=output_path)
        elif args.dataset == 'kitti':
            create_kitti_submission(args, wrapped_model, output_path=output_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', help='experiment configure file name', required=True, type=str)
    parser.add_argument('--ckpt', help='checkpoint path', required=True, type=str)
    parser.add_argument('--dataset', help='dataset to evaluate on', choices=['sintel', 'kitti', 'spring'], required=True, type=str)
    parser.add_argument('--scale', help='scale factor for input images', default=0.0, type=float)
    parser.add_argument('--output_path', help='output path', default=None, type=str)
    args = parse_args(parser)
    eval(args)


if __name__ == '__main__':
    main()
