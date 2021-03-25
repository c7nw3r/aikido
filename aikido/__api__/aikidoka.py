from abc import ABC, abstractmethod

import torch

Aikidoka = torch.nn.Module

class AikidokaResult(ABC):

    @abstractmethod
    def as_preds(self):
        pass

    @abstractmethod
    def as_probs(self, return_class_probs):
        pass
