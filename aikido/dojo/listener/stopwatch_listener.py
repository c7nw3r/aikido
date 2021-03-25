# noinspection PyUnresolvedReferences
import logging
import time

from aikido.__api__.dojo_listener import DojoListener
from aikido.__api__.listener.event.on_training_finished import OnTrainingFinished
from aikido.__api__.listener.event.on_training_started import OnTrainingStarted

logger = logging.getLogger(__name__)


class StopwatchListener(DojoListener):
    """DojoListener implementation which measures the training time."""
    stopwatch = None

    def training_started(self, event: OnTrainingStarted):
        self.stopwatch = time.time()

    def training_finished(self, event: OnTrainingFinished):
        logger.info(f"training time: {round(time.time() - self.stopwatch, 2)} seconds")
        self.stopwatch = None
