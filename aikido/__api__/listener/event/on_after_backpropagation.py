from dataclasses import dataclass

from torch import Tensor
from torch.optim import Optimizer

from aikido.__api__.aikidoka import Aikidoka
from aikido.__api__.ref import Ref
from aikido.__api__.run import Run


@dataclass
class OnAfterBackpropagation:
    aikidoka: Aikidoka
    optimizer: Optimizer
    run: Run
    loss: Ref[Tensor]