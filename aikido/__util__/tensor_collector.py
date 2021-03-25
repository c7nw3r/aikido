from typing import Optional, Union, List

import torch
from torch import Tensor
from transformers import is_torch_tpu_available
from transformers.trainer_pt_utils import DistributedTensorGatherer, nested_concat, distributed_concat, nested_numpify


class TensorCollector:
    collector: Optional[Union[Tensor, List[Tensor]]] = None

    def __init__(self, local_rank: int, num_examples: int, batch_size: Optional[int] = None):
        self.gatherer = DistributedTensorGatherer(self._get_world_size(local_rank), num_examples, batch_size)
        self.local_rank = local_rank
        self.batch_size = batch_size
        self.num_examples = num_examples

    def _get_world_size(self, local_rank: int):
        world_size = 1
        if is_torch_tpu_available():
            raise NotImplementedError()
            # world_size = xm.xrt_world_size()
        elif local_rank != -1:
            # noinspection PyUnresolvedReferences
            world_size = torch.distributed.get_world_size()
        return max(1, world_size)

    def concat(self, tensor: Tensor, repeat: bool = False, dim: int = 0):
        tensors = tensor.repeat(self.batch_size) if repeat else tensor
        self.collector = tensors if self.collector is None else torch.cat((self.collector, tensors), dim=dim)

    def concat_all(self, tensor: Union[Tensor, List[Tensor]], padding_index: int = -100):
        if type(tensor) is list and len(tensor) == 1:
            tensor = tensor[0]
        self.collector = tensor if self.collector is None else nested_concat(self.collector, tensor, padding_index)

    def gather(self):
        if self.collector is not None:
            self.gatherer.add_arrays(self._gather_and_numpify(self.collector))
            self.collector = None

    def finalize(self):
        self.gather()
        return self.gatherer.finalize()

    def _gather_and_numpify(self, tensors):
        """
        Gather value of `tensors` (tensor or list/tuple of nested tensors) and convert them to numpy before
        concatenating them to `gathered`
        """
        if tensors is None:
            return
        if is_torch_tpu_available():
            # tensors = nested_xla_mesh_reduce(tensors, name)
            raise NotImplementedError()
        elif self.local_rank != -1:
            tensors = distributed_concat(tensors)

        return nested_numpify(tensors)