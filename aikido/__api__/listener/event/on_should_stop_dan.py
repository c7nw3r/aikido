from dataclasses import dataclass

from aikido.__api__.aikidoka import Aikidoka
from aikido.__api__.dojo_kun import DojoKun


@dataclass
class OnShouldStopDan:
    aikidoka: Aikidoka
    kun: DojoKun
    dan_start: bool
