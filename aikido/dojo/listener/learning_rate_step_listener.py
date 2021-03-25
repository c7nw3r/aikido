import inspect
import logging
from dataclasses import dataclass
from importlib import import_module
from typing import Optional

# noinspection PyUnresolvedReferences
from torch.optim import Optimizer

from aikido.__api__.dojo_listener import DojoListener
from aikido.__api__.listener.event.on_after_optimization import OnAfterOptimization
from aikido.__api__.listener.event.on_kata_loaded import OnKataLoaded

logger = logging.getLogger(__name__)


@dataclass
class LearningRateStepListener(DojoListener):
    """DojoListener implementation which optimizes the learning rate via a scheduler."""
    schedule_opts: Optional[dict] = None

    def kata_load_finished(self, event: OnKataLoaded):
        if event.optimizer:
            kun = event.kun
            n_batches = len(event.data.data_loader)
            num_train_optimization_steps = int(n_batches / kun.grad_acc_steps) * kun.dans

            if self.schedule_opts and "num_training_steps" not in self.schedule_opts:
                self.schedule_opts["num_training_steps"] = num_train_optimization_steps

            # Default schedule: Linear Warmup with 10% warmup
            if self.schedule_opts is None:
                self.schedule_opts = {"name": "LinearWarmup",
                                      "num_warmup_steps": 0.1 * num_train_optimization_steps,
                                      "num_training_steps": num_train_optimization_steps}

            self.scheduler = self.get_scheduler(event.optimizer.wrapped, self.schedule_opts)

    def after_optimization(self, event: OnAfterOptimization):
        self.scheduler.step()

    def get_scheduler(self, optimizer, opts):
        """ Get the scheduler based on dictionary with options. Options are passed to the scheduler constructor."""
        schedule_name = opts.get('name')
        try:
            sched_constructor = getattr(import_module('torch.optim.lr_scheduler'), schedule_name)
        except AttributeError:
            try:
                # The method names in transformers became quite long and unhandy.
                # for convenience we offer usage of shorter alias (e.g. "LinearWarmup")
                scheduler_translations = {"LinearWarmup": "get_linear_schedule_with_warmup",
                                          "ConstantWarmup": "get_constant_schedule_with_warmup",
                                          "Constant": "get_constant_schedule",
                                          "CosineWarmup": "get_cosine_schedule_with_warmup",
                                          "CosineWarmupWithRestarts": "get_cosine_with_hard_restarts_schedule_with_warmup"
                                          }
                if schedule_name in scheduler_translations.keys():
                    schedule_name = scheduler_translations[schedule_name]
                # in contrast to torch, we actually get here a method and not a class
                sched_constructor = getattr(import_module('transformers.optimization'), schedule_name)
            except AttributeError:
                raise AttributeError(f"Scheduler '{schedule_name}' not found in 'torch' or 'transformers'")

        # get supported args of constructor
        allowed_args = inspect.signature(sched_constructor).parameters.keys()

        # convert from warmup proportion to steps if required
        if 'num_warmup_steps' in allowed_args and 'num_warmup_steps' not in opts and 'warmup_proportion' in opts:
            opts['num_warmup_steps'] = int(opts["warmup_proportion"] * opts["num_training_steps"])

        # only pass args that are supported by the constructor
        constructor_opts = {k: v for k, v in opts.items() if k in allowed_args}

        # Logging
        logger.info(f"Use scheduler `{schedule_name}`: '{constructor_opts}'")

        scheduler = sched_constructor(optimizer, **constructor_opts)
        scheduler.opts = opts  # save the opts with the scheduler to use in load/save
        return scheduler
