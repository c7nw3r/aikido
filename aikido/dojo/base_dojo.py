from dataclasses import dataclass, field
from typing import List, Callable

import torch
# noinspection PyUnresolvedReferences
from torch.optim import Optimizer
from transformers.trainer_pt_utils import nested_detach

from aikido.__api__.aikidoka import Aikidoka
from aikido.__api__.dojo import Dojo
from aikido.__api__.dojo_evaluation import DojoEvaluation
from aikido.__api__.dojo_kun import DojoKun
from aikido.__api__.dojo_listener import DojoListener
from aikido.__api__.kata import Kata, LoadedKata
from aikido.__api__.listener.dojo_listeners import DojoListeners
from aikido.__api__.listener.event import OnAfterBackpropagation, OnAfterOptimization, OnBatchFinished
from aikido.__api__.listener.event import OnBatchStarted, OnBeforeBackpropagation, OnBeforeOptimization
from aikido.__api__.listener.event import OnDanFinished, OnDanStarted, OnEvaluationFinished
from aikido.__api__.listener.event import OnEvaluationStarted, OnKataLoaded, OnShouldSkipBatch
from aikido.__api__.listener.event import OnShouldStopBatch, OnShouldStopDan, OnTrainingFinished, OnTrainingStarted
from aikido.__api__.metric import Metrics, compute_metrics
from aikido.__api__.ref import Ref
from aikido.__api__.run import Run
from aikido.__util__.tensor_collector import TensorCollector
from aikido.__util__.tensors import aggregate, backward_and_detach


@dataclass
class BaseDojo(Dojo):
    dojo_kun: DojoKun
    listeners: List[DojoListener] = field(default_factory=lambda x: [])

    def __post_init__(self):
        self.listeners = list(DojoListeners(self.listeners))

    def add_listener(self, listener: DojoListener):
        self.listeners.append(listener)

    def train(self, aikidoka: Aikidoka, kata: Kata):
        aikidoka.train()
        aikidoka = Ref(aikidoka)

        optimizer = Ref(self.dojo_kun.optimizer(aikidoka.wrapped))

        self.tell(lambda x: x.training_started(OnTrainingStarted(aikidoka, optimizer, kata, self.dojo_kun)))

        aikidoka = aikidoka.wrapped
        aikidoka.zero_grad()

        data = kata.load(self.dojo_kun.batch_size)
        # aikidoka.pre_init(data)
        self.tell(lambda x: x.kata_load_finished(OnKataLoaded(aikidoka, kata, data, self.dojo_kun, optimizer)))

        for i in range(self.dojo_kun.from_epoch, self.dojo_kun.dans):
            run = Run(i, self.dojo_kun.dans)

            if self.ask(lambda x: x.should_stop_dan(OnShouldStopDan(aikidoka, self.dojo_kun, True))):
                break

            self.tell(lambda x: x.dan_started(OnDanStarted(aikidoka, self.dojo_kun, run)))
            self.do_dan(aikidoka, data, optimizer.wrapped, run)
            self.tell(lambda x: x.dan_finished(OnDanFinished(aikidoka, self.dojo_kun, run)))

            if self.ask(lambda x: x.should_stop_dan(OnShouldStopDan(aikidoka, self.dojo_kun, False))):
                break

        # invoke training_finished listener event
        self.tell(lambda x: x.training_finished(OnTrainingFinished(aikidoka, data, self.dojo_kun, self.evaluate)))

    def do_dan(self, aikidoka: Aikidoka, data: LoadedKata, optimizer: Optimizer, run: Run):
        kun, eval_ref = self.dojo_kun, self.evaluate

        data_loader = data.data_loader
        for batch_idx, batch in enumerate(data_loader):
            run = Run(run.dan_run, run.dan_len, batch_idx, len(data_loader))
            batch_ref = Ref(batch)

            if self.ask(lambda x: x.should_skip_batch(OnShouldSkipBatch(aikidoka, self.dojo_kun, batch, run))):
                continue
            if self.ask(lambda x: x.should_stop_batch(OnShouldStopBatch(aikidoka, kun, data, run, True, eval_ref))):
                break

            self.tell(lambda x: x.batch_started(OnBatchStarted(aikidoka, self.dojo_kun, batch_ref, run)))

            batch = batch_ref.get_wrapped()
            loss_ref = Ref(aggregate(aikidoka(**batch)))

            self.tell(lambda x: x.before_backprogagation(OnBeforeBackpropagation(aikidoka, batch_ref, loss_ref, run)))

            loss_ref.apply(lambda x: x / self.dojo_kun.grad_acc_steps)
            loss_ref.apply(lambda x: backward_and_detach(x))

            self.tell(lambda x: x.after_backprogagation(OnAfterBackpropagation(aikidoka, optimizer, run, loss_ref)))

            if (batch_idx + 1) % self.dojo_kun.grad_acc_steps == 0:
                self.tell(lambda x: x.before_optimization(OnBeforeOptimization(aikidoka, batch, optimizer, run)))

                optimizer.step()
                aikidoka.zero_grad()

                self.tell(lambda x: x.after_optimization(OnAfterOptimization(aikidoka, kun, optimizer, run)))

            self.tell(lambda x: x.batch_finished(OnBatchFinished(aikidoka, batch_ref, run, loss_ref.wrapped)))

            if self.ask(lambda x: x.should_stop_batch(OnShouldStopBatch(aikidoka, kun, data, run, False, eval_ref))):
                break

    def tell(self, callback: Callable[[DojoListener], None]):
        [callback(listener) for listener in self.listeners]

    def ask(self, callback: Callable[[DojoListener], bool]):
        return True in [callback(listener) for listener in self.listeners]

    def evaluate(self, aikidoka: Aikidoka, kata: Kata, metrics: Metrics = []) -> [DojoEvaluation]:
        aikidoka.eval()
        aikidoka = Ref(aikidoka)

        # invoke training_started listener event
        self.tell(lambda x: x.evaluation_started(OnEvaluationStarted(aikidoka, kata, self.dojo_kun)))
        aikidoka = aikidoka.wrapped

        # load the kata data
        data = kata.load(self.dojo_kun.batch_size)
        # aikidoka.pre_init(data)

        # invoke kata_load_finished listener event
        self.tell(lambda x: x.kata_load_finished(OnKataLoaded(aikidoka, kata, data, self.dojo_kun)))

        loss_collector = TensorCollector(self.dojo_kun.local_rank, len(data), self.dojo_kun.batch_size)
        pred_collector = TensorCollector(self.dojo_kun.local_rank, len(data))
        label_collector = TensorCollector(self.dojo_kun.local_rank, len(data))

        with torch.no_grad():
            for batch_idx, batch in enumerate(data.data_loader):
                run = Run(0, 0, batch_idx, len(data.data_loader))
                batch_ref = Ref(batch)

                # invoke batch_started listener event
                self.tell(lambda x: x.batch_started(OnBatchStarted(aikidoka, self.dojo_kun, batch_ref, run)))
                batch = batch_ref.get_wrapped()

                result = aikidoka(**batch)
                loss = aggregate(result)

                loss_collector.concat(loss, repeat=True)
                pred_collector.concat_all(nested_detach(result[0][1]))
                # label_collector.concat_all(nested_detach(result[0][-1]))
                # pred_collector.concat_all(nested_detach(result[1]))
                label_collector.concat_all(nested_detach(batch["labels"]))  # FIXME

                # Gather all tensors and put them back on the CPU if we have done enough accumulation steps.
                if self.dojo_kun.grad_acc_steps is not None and (batch_idx + 1) % self.dojo_kun.grad_acc_steps == 0:
                    loss_collector.gather()
                    pred_collector.gather()
                    label_collector.gather()

                # invoke batch_finished listener event
                self.tell(lambda x: x.batch_finished(OnBatchFinished(aikidoka, batch, run, loss)))

        # Gather all remaining tensors and put them back on the CPU
        loss = loss_collector.finalize()
        pred = pred_collector.finalize()
        label = label_collector.finalize()

        # invoke training_finished listener event
        self.tell(lambda x: x.evaluation_finished(OnEvaluationFinished(aikidoka, kata, self.dojo_kun)))

        return DojoEvaluation(pred, loss, compute_metrics(metrics, pred, label))
