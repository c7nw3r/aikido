import logging
# noinspection PyUnresolvedReferences
from dataclasses import dataclass

import numpy as np

from aikido.__api__.dojo_listener import DojoListener
from aikido.__api__.kata import Kata
from aikido.__api__.listener.event.on_training_finished import OnTrainingFinished
from aikido.__api__.metric import Metrics

logger = logging.getLogger(__name__)


@dataclass
class ValidationListener(DojoListener):
    """DojoListener implementation which validates the aikidoka at the end of a training."""
    kata: Kata
    metrics: Metrics

    # noinspection PyUnresolvedReferences
    def training_finished(self, event: OnTrainingFinished):
        if event.kun.local_rank in [0, -1]:
            evaluation = event.evaluate(event.aikidoka, self.kata, self.metrics)

            logger.info("Evaluation (test):")
            logger.info(f"loss: {np.mean(evaluation.loss)}")
            for metric in evaluation.metrics:
                print(metric)
