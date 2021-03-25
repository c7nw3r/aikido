from dataclasses import dataclass

from torch import Tensor

from aikido.__api__.aikidoka import Aikidoka
from aikido.__api__.ref import Ref
from aikido.__api__.run import Run


@dataclass
class OnBeforeBackpropagation:
    aikidoka: Aikidoka
    batch: Ref[dict]
    loss: Ref[Tensor]
    run: Run