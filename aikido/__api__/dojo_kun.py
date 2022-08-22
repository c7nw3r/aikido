from dataclasses import dataclass
from typing import Callable

# noinspection PyUnresolvedReferences
from torch.optim import Optimizer

from aikido.__api__.aikidoka import Aikidoka


@dataclass
class DojoKun:
    dans: int = 20
    batch_size: int = 64
    max_seq_len: int = 150
    shuffle: bool = False
    local_rank: int = -1  # Local rank of process when distributed training via DDP is used
    show_progress: bool = True  # whether or not to show the progress bar
    from_step: int = 0  # the step number to start the training from when training resumes from a saved checkpoint
    device: str = "cpu"
    from_epoch: int = 0  # the epoch number to start the training from
    # Number of training steps for which the gradients should be accumulated.
    # Useful to achieve larger effective batch sizes that would not fit in GPU memory.
    grad_acc_steps: int = 1
    distributed: bool = False