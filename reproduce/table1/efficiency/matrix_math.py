"""Candidate fused arithmetic for large batches of 3D matrices and vectors."""
import torch


def matvec(matrix, vector):
    return (matrix * vector.unsqueeze(-2)).sum(dim=-1)


def transpose_matvec(matrix, vector):
    return (matrix * vector.unsqueeze(-1)).sum(dim=-2)


def matmul33(left, right):
    return (left.unsqueeze(-1) * right.unsqueeze(-3)).sum(dim=-2)


def covariance(frame, scale):
    weighted = frame * scale.square().unsqueeze(-2)
    return matmul33(weighted, frame.transpose(-1, -2))
