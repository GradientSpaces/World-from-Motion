import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

MAX_FLOW = 40000
def sequence_loss(output, flow_gt, valid, gamma=0.8, max_flow=MAX_FLOW):
    """ Loss function defined over sequence of flow predictions """
    n_predictions = len(output['flow'])
    flow_loss = 0.0
    # exlude invalid pixels and extremely large diplacements
    mag = torch.sum(flow_gt**2, dim=1).sqrt()
    valid = (valid >= 0.5) & (mag < max_flow)
    for i in range(n_predictions):
        i_weight = gamma ** (n_predictions - i - 1)
        loss_i = output['nf'][i]
        mask = (~torch.isnan(loss_i.detach())) & (~torch.isinf(loss_i.detach())) & valid[:, None]
        if mask.sum() == 0:
            flow_loss += 0 * loss_i.sum()
        else:
            flow_loss += i_weight * ((mask * loss_i).sum()) / mask.sum()

    return flow_loss