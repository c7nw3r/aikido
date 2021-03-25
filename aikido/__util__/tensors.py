from typing import Union, List, Tuple

import torch
from torch import Tensor


def to_device(batch: dict, device: str) -> dict:
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            batch[k] = v.to(device)

    return batch


def aggregate(tensors: Union[Tensor, List], dim=0) -> Tensor:
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