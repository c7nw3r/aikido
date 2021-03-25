from abc import ABC

from torch.nn import Module
from transformers.modeling_outputs import BaseModelOutputWithPoolingAndCrossAttentions


class AbstractTransformerPooling(Module, ABC):

    def forward(self, x: BaseModelOutputWithPoolingAndCrossAttentions):
        pass
