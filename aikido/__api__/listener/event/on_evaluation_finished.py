from dataclasses import dataclass

from aikido.__api__.aikidoka import Aikidoka
from aikido.__api__.dojo_kun import DojoKun
from aikido.__api__.kata import Kata


@dataclass
class OnEvaluationFinished:
    aikidoka: Aikidoka
    kata: Kata
    kun: DojoKun