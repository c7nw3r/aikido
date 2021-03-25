from dataclasses import dataclass

from torch.optim import Optimizer

from aikido.__api__.aikidoka import Aikidoka
from aikido.__api__.dojo_kun import DojoKun
from aikido.__api__.run import Run


@dataclass
class OnAfterOptimization:
    aikidoka: Aikidoka
    kun: DojoKun
    optimizer: Optimizer
    run: Run