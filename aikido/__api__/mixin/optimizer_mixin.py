import torch.optim as optim
from torch.optim import Optimizer, lr_scheduler

from aikido.__api__.aikidoka import Aikidoka


class OptimizerMixin:

    def get_optimizer(self, aikidoka: Aikidoka) -> Optimizer:
        return optim.Adam(aikidoka.parameters(), lr=1e-3)

    def get_scheduler(self, optimizer: Optimizer):
        return lr_scheduler.StepLR(optimizer, 8, gamma=0.1, last_epoch=-1)