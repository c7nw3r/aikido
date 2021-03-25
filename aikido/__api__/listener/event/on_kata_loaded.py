from dataclasses import dataclass
from typing import Optional

# noinspection PyUnresolvedReferences
from torch.optim import Optimizer

from aikido.__api__.aikidoka import Aikidoka
from aikido.__api__.dojo_kun import DojoKun
from aikido.__api__.kata import Kata, LoadedKata
from aikido.__api__.ref import Ref


@dataclass
class OnKataLoaded:
    aikidoka: Ref[Aikidoka]
    kata: Kata
    data: LoadedKata
    kun: DojoKun
    optimizer: Optional[Ref[Optimizer]] = None
