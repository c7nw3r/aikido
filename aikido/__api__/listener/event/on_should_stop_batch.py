from dataclasses import dataclass

from aikido.__api__.aikidoka import Aikidoka
from aikido.__api__.dojo_kun import DojoKun
from aikido.__api__.kata import LoadedKata
from aikido.__api__.run import Run
from aikido.__api__.trait.dojo.evaluation_trait import Eval


@dataclass
class OnShouldStopBatch(object):
    aikidoka: Aikidoka
    kun: DojoKun
    data: LoadedKata
    run: Run
    batch_start: bool
    evaluate: Eval
