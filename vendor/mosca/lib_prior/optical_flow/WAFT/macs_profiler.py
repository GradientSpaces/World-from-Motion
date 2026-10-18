import sys
import argparse
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data

from model import fetch_model
from config.parser import parse_args

class NetWrapper(nn.Module):
    def __init__(self, args):
        super(NetWrapper, self).__init__()
        self.model = fetch_model(args)

    def forward(self, x):
        return self.model(x, x)

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def eval(args):
    args.gpus = [0]
    model = NetWrapper(args).cuda()
    model.eval()
    input = torch.randn(1, 3, 540, 960).cuda()
    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA
        ],
        with_flops=True) as prof:
            output = model(input)
    
    print(prof.key_averages(group_by_stack_n=5).table(sort_by='self_cuda_time_total', row_limit=5))
    events = prof.events()
    forward_MACs = sum([int(evt.flops) for evt in events])
    print("forward MACs: ", forward_MACs / 2 / 1e9, "G")

    print("Number of parameters: ", count_parameters(model) / 1e6, "M")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', help='experiment configure file name', required=True, type=str) 
    args = parse_args(parser)
    eval(args)

if __name__ == '__main__':
    main()

