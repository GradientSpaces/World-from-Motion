"""Fusible covariance and spherical-harmonic preparation for the renderer."""
import torch
import torch.nn.functional as F
from functools import lru_cache
import threading

from lib_render.sh_utils import eval_sh
from efficiency.matrix_math import covariance, transpose_matvec

_compilation_entry = threading.RLock()


def prepare_gaussians(xyz, frame, scale, color_feat, degree):
    cov = covariance(frame, scale)
    direction = F.normalize(xyz, dim=-1)
    local_direction = F.normalize(transpose_matvec(frame, direction), dim=-1)
    sh = color_feat.reshape(color_feat.shape[0], -1, 3).permute(0, 2, 1)
    color = eval_sh(degree, sh, local_direction)
    return cov, color


def _degree_three(xyz, frame, scale, color_feat):
    return prepare_gaussians(xyz, frame, scale, color_feat, 3)


@lru_cache(maxsize=1)
def compiled_degree_three():
    # Degree zero hits a PyTorch 2.3 code-generation issue; callers retain the
    # existing eager path for all degrees other than the measured degree three.
    compiled = torch.compile(_degree_three, fullgraph=True, dynamic=True)

    def call(*parameters):
        # WorldTree renders concurrently on two devices. PyTorch 2.3's FX
        # tracing flag is global, and a second caller can otherwise enter
        # Dynamo while the first caller compiles a new device/gradient graph.
        # Serialize entry/compilation, without synchronizing GPU execution.
        with _compilation_entry:
            return compiled(*parameters)

    return call


def reference_gaussians(xyz, frame, scale, color_feat, degree):
    diagonal = torch.zeros_like(frame)
    diagonal[:, 0, 0] = scale[:, 0]
    diagonal[:, 1, 1] = scale[:, 1]
    diagonal[:, 2, 2] = scale[:, 2]
    cov = frame @ (diagonal ** 2) @ frame.permute(0, 2, 1)
    direction = F.normalize(xyz, dim=-1)
    local_direction = torch.einsum("nji,nj->ni", frame, direction)
    local_direction = F.normalize(local_direction, dim=-1)
    sh = color_feat.reshape(color_feat.shape[0], -1, 3).permute(0, 2, 1)
    return cov, eval_sh(degree, sh, local_direction)
