from typing import Optional

import torch
from torch.nn import Module
from transformers import PretrainedConfig


class PredictionHead(Module):

    def init(self, config: PretrainedConfig):
        pass

    def get_buffer(self, name: str) -> Optional[torch.Tensor]:
        for buffer in self.named_buffers():
            if buffer[0] == name:
                return buffer[1]
        return None