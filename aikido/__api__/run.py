from dataclasses import dataclass
from typing import Optional


@dataclass
class Run(object):
    dan_run: int
    dan_len: int
    batch_run: Optional[int] = None
    batch_len: Optional[int] = None