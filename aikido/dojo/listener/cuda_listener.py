import logging
import warnings
from dataclasses import dataclass

import torch
from torch import cuda
# noinspection PyUnresolvedReferences
from torch.optim import Optimizer

from aikido.__api__.dojo_listener import DojoListener
from aikido.__api__.listener.event import OnBatchStarted, OnBeforeBackpropagation
from aikido.__api__.listener.event.on_evaluation_started import OnEvaluationStarted
from aikido.__api__.listener.event.on_training_started import OnTrainingStarted
from aikido.__util__.tensors import to_device


@dataclass
class CudaListener(DojoListener):
    """DojoListener implementation which enables CUDA if available."""
    reset_memory: bool = False
    ignore_future_warning: bool = True

    def training_started(self, event: OnTrainingStarted):
        if cuda.is_available():
            if self.reset_memory:
                if self.ignore_future_warning:
                    warnings.simplefilter(action="ignore", category=FutureWarning)
                torch.cuda.reset_max_memory_allocated()
                torch.cuda.reset_max_memory_cached()

            logging.info("enable CUDA for aikidoka")
            event.aikidoka.to("cuda")
            event.kun.device = "cuda"

    def evaluation_started(self, event: OnEvaluationStarted):
        self.training_started(OnTrainingStarted(event.aikidoka, event.kata, event.kun))

    def batch_started(self, event: OnBatchStarted):
        to_device(event.batch.wrapped, event.kun.device)

    def before_backprogagation(self, event: OnBeforeBackpropagation):
        to_device(event.batch.wrapped, "cpu")

    def get_order(self) -> int:
        return -100
