# noinspection PyUnresolvedReferences
from torch.optim import Optimizer
from tqdm import tqdm

from aikido.__api__.dojo_listener import DojoListener
from aikido.__api__.listener.event import OnEvaluationFinished
from aikido.__api__.listener.event.on_batch_finished import OnBatchFinished
from aikido.__api__.listener.event.on_dan_finished import OnDanFinished
from aikido.__api__.listener.event.on_dan_started import OnDanStarted
from aikido.__api__.listener.event.on_evaluation_started import OnEvaluationStarted
from aikido.__api__.listener.event.on_training_finished import OnTrainingFinished
from aikido.__api__.listener.event.on_training_started import OnTrainingStarted


class TqdmListener(DojoListener):
    """DojoListener implementation which displays a tqdm progress bar."""

    def __init__(self):
        self.progress = None
        self.dataset_name = []

    def training_started(self, event: OnTrainingStarted):
        self.dataset_name.append("train")

    def evaluation_started(self, event: OnEvaluationStarted):
        self.dataset_name.append("test")

    def evaluation_finished(self, event: OnEvaluationFinished):
        self.dataset_name.pop()

    def training_finished(self, event: OnTrainingFinished):
        self.dataset_name.pop()

    def dan_started(self, e: OnDanStarted):
        disabled = e.kun.local_rank not in [0, -1] or e.kun.show_progress is False
        self.progress = tqdm(total=e.kun.dans, desc="init ...", disable=disabled, position=0, leave=True, unit=" runs")

    def batch_finished(self, event: OnBatchFinished):
        run = event.run

        dan_run = f"{run.dan_run + 1}/{run.dan_len}"
        batch_run = f"{run.batch_run + 1}/{run.batch_len}"

        if self.dataset_name[-1] == "test":
            desc = f"Run (test) batch({batch_run})"
        else:
            desc = f"Run (train) dan({dan_run}) batch({batch_run}) loss({round(event.loss.mean().item(), 4)})"

        # FIXME
        if self.progress:
            self.progress.set_description(desc)
            self.progress.update()

    def dan_finished(self, event: OnDanFinished):
        self.progress.close()
