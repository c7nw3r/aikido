from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Union, Dict

# import datasets
import numpy as np
# from datasets import load_metric

Metrics = Union[List['Metric'], Dict[int, List['Metric']]]

@dataclass
class MetricRequest:
    preds: Union[np.ndarray, List]
    refs: Union[np.ndarray, List]


def compute_metrics(metrics: Metrics, pred, label):
    metric_list = []
    if type(metrics) is list:
        for metric in metrics:
            metric_list.append(metric(MetricRequest(pred, label)))
    else:
        for index, metric in metrics:
            # FIXME: handle non list pred/label
            metric_list.append(metric(MetricRequest(pred[index], label[index])))

    return metric_list


class Metric(ABC):

    @staticmethod
    def load(name: str) -> 'Metric':
        # return HuggingFaceMetric(load_metric(name))
        return None

    @abstractmethod
    def __call__(self, request: MetricRequest) -> dict:
        pass


# @dataclass
# class HuggingFaceMetric(Metric):
#     metric: datasets.Metric
#
#     def __call__(self, request: MetricRequest) -> dict:
#         return self.metric.compute(request.preds, request.refs)

