import copy
import logging
from typing import List

import numpy as np
import torch
from sklearn.utils import compute_class_weight
from torch import nn
from torch.nn import CrossEntropyLoss

from aikido.__api__.kata import LoadedKata
from aikido.modeling.nn.head import PredictionHead
from aikido.modeling.nn.head.result.token_classifier_result import TokenClassifierResult

logger = logging.getLogger(__name__)


class TokenClassifierHead(PredictionHead):
    def __init__(self, labels: List[str], label_name: str = "labels", embeds_dropout_prob: float = 0.1):
        super(TokenClassifierHead, self).__init__()
        self.labels = labels
        self.label_name = label_name

        self.dropout = nn.Dropout(embeds_dropout_prob)
        self.feed_forward = nn.Linear(768, len(labels))

    def pre_init(self, data: LoadedKata):
        # initialize the array with the labels to avoid zero division
        observed_labels = copy.deepcopy(self.labels)
        for x in data.data_loader:
            observed_labels += [self.labels[label_id] for label_id in x[self.label_name]]

        weights = compute_class_weight("balanced", classes=np.asarray(self.labels), y=observed_labels)
        self.register_buffer("balance", torch.tensor(weights.astype(np.float32)))

    def forward(self, outputs, batch):
        return_dict = batch["return_dict"] if "return_dict" in batch else False

        sequence_output = outputs[0]
        sequence_output = self.dropout(sequence_output)
        logits = self.feed_forward(sequence_output)

        loss = self._calculate_loss(logits, batch)

        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return TokenClassifierResult(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def _calculate_loss(self, logits, batch):
        weights = self.get_buffer("balance")
        weight = nn.Parameter(weights, requires_grad=False)
        weight = weight if weight.shape[0] > 0 else None
        if weight is not None:
            weight = weight.to("cuda")  # FIXME

        loss_fct = CrossEntropyLoss(weight=weight)
        loss = None
        if self.label_name in batch:
            labels = batch["labels"]
            # Only keep active parts of the loss
            if "attention_mask" in batch:
                active_loss = batch["attention_mask"].view(-1) == 1
                active_logits = logits.view(-1, len(self.labels))
                active_labels = torch.where(
                    active_loss, batch["labels"].view(-1), torch.tensor(loss_fct.ignore_index).type_as(labels)
                )
                loss = loss_fct(active_logits, active_labels)
            else:
                loss = loss_fct(logits.view(-1, len(self.labels)), labels.view(-1))
        return loss

