from dataclasses import dataclass, field

import torch
# noinspection PyUnresolvedReferences
from torch.optim import Optimizer

from aikido.__api__.dojo_listener import DojoListener
from aikido.__api__.listener.event.on_before_optimization import OnBeforeOptimization


@dataclass
class GradientClippingListener(DojoListener):
    max_grad_norm: float = field(default=1.0, metadata={"desc": "Max gradient norm."})

    def __post_init__(self):
        assert self.max_grad_norm > 0, "max_grad_norm must be greater than zero"

    def before_optimization(self, event: OnBeforeOptimization):
        if hasattr(event.optimizer, "clip_grad_norm"):
            # Some optimizers (like the sharded optimizer) have a specific way to do gradient clipping
            assert callable(event.optimizer.clip_grad_norm)
            event.optimizer.clip_grad_norm(self.max_grad_norm)
        else:
            # Revert to normal clipping otherwise, handling Apex or full precision
            torch.nn.utils.clip_grad_norm_(
                # amp.master_params(self.optimizer) if self.use_apex else model.parameters(),
                event.aikidoka.parameters(),
                self.max_grad_norm,
            )
