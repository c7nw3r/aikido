from dataclasses import dataclass

from aikido.__api__.aikidoka import Aikidoka
from aikido.__api__.dojo_kun import DojoKun
from aikido.__api__.ref import Ref
from aikido.__api__.run import Run


@dataclass
class OnBatchStarted:
    aikidoka: Aikidoka
    kun: DojoKun
    batch: Ref[dict]
    run: Run