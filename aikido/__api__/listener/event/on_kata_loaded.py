from dataclasses import dataclass

# noinspection PyUnresolvedReferences
from torch.optim import Optimizer
from torch.utils.data import DataLoader

from aikido.__api__.aikidoka import Aikidoka
from aikido.__api__.dojo_kun import DojoKun
from aikido.__api__.kata import Kata


@dataclass
class OnKataLoaded:
    aikidoka: Aikidoka
    kata: Kata
    data: DataLoader
    kun: DojoKun
