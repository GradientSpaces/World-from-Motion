import logging
import socket
import argparse
from datetime import datetime, timedelta

import torch
import os
import torch.nn as nn
import torch.optim as optim
from torch.autograd.profiler import record_function

from model import fetch_model
from criterion.loss import sequence_loss
from config.parser import parse_args

PREFIX = "../memory_profile_results/"

def fetch_optimizer(args, model):
    """ Create the optimizer and learning rate scheduler """
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wdecay, eps=args.epsilon)
    scheduler = optim.lr_scheduler.OneCycleLR(optimizer, args.lr, args.num_steps + 100,
        pct_start=0.05, cycle_momentum=False, anneal_strategy='linear')

    return optimizer, scheduler

class NetWrapper(nn.Module):
    def __init__(self, args):
        super(NetWrapper, self).__init__()
        self.model = fetch_model(args)

    def forward(self, x, flow_gt=None):
        return self.model(x, x, flow_gt=flow_gt)

logging.basicConfig(
   format="%(levelname)s:%(asctime)s %(message)s",
   level=logging.INFO,
   datefmt="%Y-%m-%d %H:%M:%S",
)
logger: logging.Logger = logging.getLogger(__name__)
logger.setLevel(level=logging.INFO)

TIME_FORMAT_STR: str = "%b_%d_%H_%M_%S"

def trace_handler(prof: torch.profiler.profile):
    # Prefix for file names.
    host_name = socket.gethostname()
    timestamp = datetime.now().strftime(TIME_FORMAT_STR)
    file_prefix = f"{PREFIX}/{host_name}_{timestamp}"
    os.makedirs(PREFIX, exist_ok=True)
    # Construct the trace file.
    prof.export_chrome_trace(f"{file_prefix}.json.gz")
    # Construct the memory timeline file.
    prof.export_memory_timeline(f"{file_prefix}.html", device="cuda:0")
    print(f"Memory timeline saved to {file_prefix}.html")

def run_model(args, num_iters=5, device="cuda:0"):
    global PREFIX
    PREFIX += args.algorithm
    model = NetWrapper(args).cuda()
    inputs = torch.randn(1, 3, 544, 960).cuda()
    optimizer, scheduler = fetch_optimizer(args, model)
    flow = torch.randn(1, 2, 544, 960).cuda()
    valid = torch.ones(1, 1, 544, 960).cuda()
    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ],
        schedule=torch.profiler.schedule(wait=0, warmup=0, active=6, repeat=1),
        record_shapes=True,
        profile_memory=True,
        with_stack=True,
        on_trace_ready=trace_handler,
    ) as prof:
        for _ in range(num_iters):
            prof.step()
            with record_function("## forward ##"):
                output = model(inputs, flow_gt=flow)
            with record_function("## backward ##"):
                loss = sequence_loss(output, flow, valid, args.gamma)
                loss.backward()
            with record_function("## optimizer ##"):
                optimizer.step()
            optimizer.zero_grad(set_to_none=True)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', help='experiment configure file name', required=True, type=str) 
    args = parse_args(parser)
    run_model(args)

if __name__ == "__main__":
    main()