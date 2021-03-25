# noinspection PyUnresolvedReferences
from torch.optim import Optimizer

from aikido.__api__.listener.event.on_after_backpropagation import OnAfterBackpropagation
from aikido.__api__.listener.event.on_after_optimization import OnAfterOptimization
from aikido.__api__.listener.event.on_batch_finished import OnBatchFinished
from aikido.__api__.listener.event.on_batch_started import OnBatchStarted
from aikido.__api__.listener.event.on_before_backpropagation import OnBeforeBackpropagation
from aikido.__api__.listener.event.on_before_optimization import OnBeforeOptimization
from aikido.__api__.listener.event.on_dan_finished import OnDanFinished
from aikido.__api__.listener.event.on_dan_started import OnDanStarted
from aikido.__api__.listener.event.on_evaluation_finished import OnEvaluationFinished
from aikido.__api__.listener.event.on_evaluation_started import OnEvaluationStarted
from aikido.__api__.listener.event.on_kata_loaded import OnKataLoaded
from aikido.__api__.listener.event.on_should_skip_batch import OnShouldSkipBatch
from aikido.__api__.listener.event.on_should_stop_batch import OnShouldStopBatch
from aikido.__api__.listener.event.on_should_stop_dan import OnShouldStopDan
from aikido.__api__.listener.event.on_training_finished import OnTrainingFinished
from aikido.__api__.listener.event.on_training_started import OnTrainingStarted


class DojoListener:
    """
    Instances which extends from this class can be used to listen to certain dojo events.
    """

    def kata_load_finished(self, event: OnKataLoaded):
        pass

    def dan_started(self, event: OnDanStarted):
        pass

    def dan_finished(self, event: OnDanFinished):
        pass

    def training_started(self, event: OnTrainingStarted):
        pass

    def training_finished(self, event: OnTrainingFinished):
        pass

    def evaluation_started(self, event: OnEvaluationStarted):
        pass

    def evaluation_finished(self, event: OnEvaluationFinished):
        pass

    def batch_started(self, event: OnBatchStarted):
        pass

    def batch_finished(self, event: OnBatchFinished):
        pass

    def before_backprogagation(self, event: OnBeforeBackpropagation):
        pass

    def after_backprogagation(self, event: OnAfterBackpropagation):
        pass

    def before_optimization(self, event: OnBeforeOptimization):
        pass

    def after_optimization(self, event: OnAfterOptimization):
        pass

    def should_skip_batch(self, _event: OnShouldSkipBatch) -> bool:
        return False

    def should_stop_batch(self, event: OnShouldStopBatch) -> bool:
        return False

    def should_stop_dan(self, event: OnShouldStopDan) -> bool:
        return False

    def get_order(self) -> int:
        return 0
