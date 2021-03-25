import logging
from dataclasses import dataclass

import numpy as np
# noinspection PyUnresolvedReferences
from torch.optim import Optimizer

from aikido.__api__.dojo_evaluation import DojoEvaluation
from aikido.__api__.dojo_listener import DojoListener
from aikido.__api__.kata import Kata
from aikido.__api__.listener.event import OnShouldStopDan
from aikido.__api__.listener.event.on_should_stop_batch import OnShouldStopBatch
from aikido.__api__.listener.event.on_training_started import OnTrainingStarted
from aikido.__api__.metric import Metrics
from aikido.__api__.run import Run

logger = logging.getLogger(__name__)


@dataclass
class EvaluationListener(DojoListener):
    """
    Can be used to control early stopping with a Dojo implementation. Any object can be used instead which
    implements the method check_stopping and and provides the attribute save_dir
    """

    kata: Kata
    metrics: Metrics
    save_dir = None  # the directory where to save the final best model, if None, no saving
    mode: str = "min"  # "min" or "max"
    patience: int = 0  # how many evaluations to wait after the best evaluation to stop
    min_delta: float = 0.001  # minimum difference to a previous best value to count as an improvement
    min_evals: int = 0  # minimum number of evaluations to wait before using eval value
    evaluate_every: int = 100  # Perform dev set evaluation after this many steps of training
    early_stopping: bool = False

    def __post_init__(self):
        self.eval_values = []  # for more complex modes
        self.n_since_best = None
        self.do_stopping = False
        if self.mode == "min":
            self.best_so_far = 1.0E99
        elif self.mode == "max":
            self.best_so_far = -1.0E99
        else:
            raise Exception("Mode must be 'min' or 'max'")

    def training_started(self, event: OnTrainingStarted):
        self.do_stopping = False
        self.eval_values = []

    def should_stop_dan(self, event: OnShouldStopDan) -> bool:
        return not event.dan_start and self.do_stopping

    def should_stop_batch(self, event: OnShouldStopBatch) -> bool:
        if not event.batch_start:
            return False

        global_step = (event.run.dan_run * event.run.batch_len) + event.run.batch_run

        if self.evaluate_every != 0 \
                and global_step % self.evaluate_every == 0 \
                and global_step != 0 \
                and event.kun.local_rank in [0, -1]:

            evaluation = event.evaluate(event.aikidoka, self.kata, self.metrics)
            event.aikidoka.train()

            logger.info("Evaluation (dev):")
            logger.info(f"loss: {np.mean(evaluation.loss)}")
            for metric in evaluation.metrics:
                print(metric)

            if not self.early_stopping:
                return False

            self.do_stopping, save_model = self.should_stop_or_save(evaluation, event.run)
            if save_model:
                logger.info("Saving current best model to {}".format(self.save_dir))
                self.model.save(self.save_dir)
                # self.data_silo.processor.save(self.early_stopping.save_dir)
            if self.do_stopping:
                logger.info("STOPPING EARLY AT EPOCH {}, STEP {}".format(event.run.dan_run, global_step))
        return self.do_stopping

    def should_stop_or_save(self, evaluation: DojoEvaluation, run: Run):
        loss = np.mean(evaluation.loss)
        self.eval_values.append(loss)

        stop, save = False, False
        if len(self.eval_values) <= self.min_evals:
            return stop, save
        if self.mode == "min":
            delta = self.best_so_far - loss
        else:
            delta = loss - self.best_so_far
        if delta > self.min_delta:
            self.best_so_far = loss
            self.n_since_best = 0
            if self.save_dir:
                save = True
        else:
            self.n_since_best += 1
        if self.n_since_best > self.patience and run.dan_run > 0:
            stop = True
        return stop, save

    def get_order(self) -> int:
        return 1000
