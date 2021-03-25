import logging
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np
import torch
from torch import Tensor
from torch.nn import Sigmoid
from transformers.modeling_outputs import SequenceClassifierOutput

from aikido.__api__.aikidoka import AikidokaResult

logger = logging.getLogger(__name__)


@dataclass
class SequenceClassifierResult(SequenceClassifierOutput, AikidokaResult):
    labels: Optional[Tensor] = None
    label_names: List[str] = field(default_factory=lambda x: [])
    pred_threshold: float = 0.5

    def as_probs(self, return_class_probs: bool = True):
        if len(self.label_names) > 2:
            return Sigmoid()(self.logits).cpu().numpy()

        probs = torch.nn.Softmax(dim=1)(self.logits)
        return (probs if return_class_probs else torch.max(probs, dim=1)[0]).cpu().numpy()

    def as_preds(self):
        if len(self.label_names) > 2:
            pred_ids = [np.where(row > self.pred_threshold)[0] for row in self.as_probs()]
            preds = []
            for row in pred_ids:
                preds.append([self.label_names[int(x)] for x in row])

            return preds

        return [self.label_names[int(x)] for x in self.logits.cpu().numpy().argmax(1)]
