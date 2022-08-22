from typing import Union, List, Tuple, Dict

import torch
from torch import Tensor

from aikido.__util__.value_dict import ValueDict


def to_device(batch: Union[dict, list], device: str):
    if isinstance(batch, list):
        [to_device(e, device) for e in batch]

    else:
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                batch[k] = v.to(device)


def aggregate(tensors: Union[Tensor, List, Dict], dim=0) -> Tensor:
    if type(tensors) is dict:
        return tensors["loss"]
    if isinstance(tensors, ValueDict):
        return tensors["loss"]
    if type(tensors) is list:
        return torch.stack([tensor[dim] for tensor in tensors]).mean()
    return tensors[dim].mean()


def nested_detach(tensors: Union[Tuple, List], dim: int):
    "Detach `tensors` (even if it's a nested list/tuple of tensors)."
    if isinstance(tensors, (list, tuple)):
        return type(tensors)(nested_detach(t[dim], dim) for t in tensors)
    return tensors[dim].detach()

def backward_and_detach(tensor: Tensor):
    tensor.backward()
    return tensor.detach().cpu()

def to_tensor(x):
    if not isinstance(x, Tensor):
        return torch.tensor(x)
    return x

def unsqueeze4d(x):
    if x.dim() == 3:
        x = x.unsqueeze(dim=0)
    return x

import numpy as np
from numpy import matmul
from numpy.linalg import norm

def cosine_similarity(a, b):
    a = np.expand_dims(a, axis=0) if len(a.shape) == 1 else a
    b = np.expand_dims(b, axis=0) if len(b.shape) == 1 else b
    a_norm = a / norm(a, ord=2, axis=1, keepdims=True)
    b_norm = b / norm(b, ord=2, axis=1, keepdims=True)

    return np.ravel(matmul(a_norm, b_norm.transpose())).item()