import logging
from dataclasses import dataclass

import torch
# noinspection PyUnresolvedReferences
from torch.optim import Optimizer

from aikido.__api__.dojo_listener import DojoListener
from aikido.__api__.listener.event.on_evaluation_started import OnEvaluationStarted
from aikido.__api__.listener.event.on_training_started import OnTrainingStarted


def set_all_seeds(seed_value):
    logging.info("set seed to " + str(seed_value))

    from random import seed
    seed(seed_value)

    torch.manual_seed(seed_value)
    # noinspection PyUnresolvedReferences
    torch.cuda.manual_seed_all(seed_value)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    import numpy as np
    np.random.seed(seed_value)

    import os
    os.environ['PYTHONHASHSEED'] = str(seed_value)

@dataclass
class SeedListener(DojoListener):
    """DojoListener implementation which initializes all random value generators with a seed."""
    seed: int

    def evaluation_started(self, event: OnEvaluationStarted):
        set_all_seeds(self.seed)

    def training_started(self, event: OnTrainingStarted):
        set_all_seeds(self.seed)

    def get_order(self) -> int:
        return -10000

