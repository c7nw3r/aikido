from dataclasses import dataclass

from torch.optim import Optimizer

from aikido.__api__.aikidoka import Aikidoka
from aikido.__api__.dojo_kun import DojoKun
from aikido.__api__.kata import Kata
from aikido.__api__.ref import Ref


@dataclass
class OnTrainingStarted:
    aikidoka: Aikidoka
    kata: Kata
    kun: DojoKun