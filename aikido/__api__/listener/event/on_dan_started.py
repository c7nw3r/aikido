from dataclasses import dataclass

from aikido.__api__.aikidoka import Aikidoka
from aikido.__api__.dojo_kun import DojoKun
from aikido.__api__.run import Run


@dataclass
class OnDanStarted:
    aikidoka: Aikidoka
    kun: DojoKun
    run: Run