from dataclasses import dataclass
from typing import Optional

from torch import Tensor

from aikido.__api__.aikidoka import Aikidoka
from aikido.__api__.ref import Ref
from aikido.__api__.run import Run


@dataclass
class OnBatchFinished:
    aikidoka: Aikidoka
    batch: Ref[dict]
    run: Run
    loss: Optional[Tensor]
