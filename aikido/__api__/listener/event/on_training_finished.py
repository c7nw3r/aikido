from dataclasses import dataclass, field

from torch.utils.data import DataLoader

from aikido.__api__.aikidoka import Aikidoka
from aikido.__api__.dojo_kun import DojoKun
from aikido.__api__.trait.dojo.evaluation_trait import Eval


@dataclass
class OnTrainingFinished:
    aikidoka: Aikidoka = field(metadata={"desc": "The trained aikidoka"})
    data: DataLoader = field(metadata={"desc": "The training kata"})
    kun: DojoKun = field(metadata={"desc": "The dojo kun instance"})
    evaluate: Eval = field(metadata={"desc": "Evaluation callback function"})
