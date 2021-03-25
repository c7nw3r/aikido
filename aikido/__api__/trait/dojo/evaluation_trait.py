from typing import Callable

from aikido.__api__.aikidoka import Aikidoka
from aikido.__api__.dojo_evaluation import DojoEvaluation
from aikido.__api__.kata import Kata
from aikido.__api__.metric import Metrics

Eval = Callable[[Aikidoka, Kata, Metrics], DojoEvaluation]


class EvaluationTrait(object):

    def evaluate(self, aikidoka: Aikidoka, kata: Kata, metrics: Metrics = []) -> [DojoEvaluation]:
        """
        Evaluates the given aikidoka with the given kata. Returns a tuple containing the expected and predicted labels.
        Merges predictions with the same identifier column if the "merge" attribute is set to True.
        """
        pass
