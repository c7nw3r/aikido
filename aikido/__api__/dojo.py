# noinspection PyUnresolvedReferences
from torch.optim import Optimizer

from aikido.__api__.aikidoka import Aikidoka
from aikido.__api__.dojo_listener import DojoListener
from aikido.__api__.kata import Kata
from aikido.__api__.trait.dojo.evaluation_trait import EvaluationTrait


class Dojo(EvaluationTrait):

    def add_listener(self, listener: DojoListener):
        pass

    def train(self, aikidoka: Aikidoka, kata: Kata):
        """
        Trains the given aikidoka with the given kata.
        If you want to get more detailed information about the training progress
        register a proper listener instance.
        """
        pass
