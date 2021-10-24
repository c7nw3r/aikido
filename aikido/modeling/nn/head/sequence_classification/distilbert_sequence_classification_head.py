import copy
import logging
from typing import List

import numpy as np
import torch
from sklearn.utils import compute_class_weight
from torch import nn
from torch.nn import CrossEntropyLoss, MSELoss, BCEWithLogitsLoss
from transformers import ElectraConfig, PretrainedConfig, BertConfig, BertForSequenceClassification
from transformers.models.electra.modeling_electra import ElectraClassificationHead, ElectraForSequenceClassification

from aikido.__api__.kata import LoadedKata
from aikido.__util__.attribute_dict import AttributeDict
from aikido.modeling.nn.head import PredictionHead
from aikido.modeling.nn.head.result.sequence_classifier_result import SequenceClassifierResult

logger = logging.getLogger(__name__)


class DistilbertSequenceClassificationHead(PredictionHead):
    def __init__(self,
                 labels: List[str],
                 label_name="labels",
                 loss_ignore_index: int = -100,
                 loss_reduction: str = "none",
                 embeds_dropout_prob: float = 0.1,
                 multiclass: bool = False):
        super(DistilbertSequenceClassificationHead, self).__init__()
        self.labels = labels
        self.label_name = label_name
        self.dropout = nn.Dropout(embeds_dropout_prob)
        self.classifier = nn.Linear(768, len(self.labels))
        self.loss_ignore_index = loss_ignore_index
        self.loss_reduction = loss_reduction
        self.multiclass = multiclass

    def init(self, config: PretrainedConfig):
        if type(config) is ElectraConfig:
            _config = AttributeDict({"hidden_size": 768, "hidden_dropout_prob": 0.1, "num_labels": len(self.labels)})
            self.classifier = ElectraClassificationHead(_config)
        # if type(config) is BertConfig:
        #     _config = AttributeDict({"hidden_size": 768, "hidden_dropout_prob": 0.1, "num_labels": len(self.labels)})
        #     self.classifier = ElectraClassificationHead(_config)
        else:
            raise ValueError(f"cannot handle config {type(config)}")

    def init_balance(self, data: LoadedKata):
        # initialize the array with the labels to avoid zero division
        observed_labels = copy.deepcopy(self.labels)

        if not self.multiclass:  # len(self.labels) <= 2:
            for x in data.data_loader:
                observed_labels += list(map(lambda z: self.labels[z], x[self.label_name].tolist()))
        else:
            for x in data.data_loader:
                observed_labels += [self.labels[index] for _, index in (x[self.label_name] == 1).nonzero()]

        weights = compute_class_weight("balanced", classes=np.asarray(self.labels), y=observed_labels)
        weights = torch.tensor(weights.astype(np.float32))
        self.register_buffer("balance", weights)

    def forward(self, outputs, batch):
        return_dict = batch["return_dict"] if "return_dict" in batch else False

        pooled_output = self.dropout(outputs[1 if len(outputs) > 1 else 0])
        logits = self.classifier(pooled_output)

        loss = self._calculate_loss(logits, batch)
        labels = batch[self.label_name] if self.label_name in batch else None

        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output + (labels,)) if loss is not None else output

        return SequenceClassifierResult(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            label_names=self.labels,
            labels=labels
        )

    def _calculate_loss(self, logits, batch):
        weights = self.get_buffer("balance")
        weight = nn.Parameter(weights, requires_grad=False)
        weight = weight if weight.shape[0] > 0 else None
        if weight is not None:
            weight = weight.to("cuda")  # FIXME

        labels = batch[self.label_name]

        loss = None
        if labels is not None:
            if len(self.labels) == 1:
                #  We are doing regression
                loss_fct = MSELoss()
                loss = loss_fct(logits.view(-1), labels.view(-1))
            elif not self.multiclass:
                loss_fct = CrossEntropyLoss(weight=weight)
                loss = loss_fct(logits.view(-1, len(self.labels)), labels.view(-1))
            else:
                labels = labels.to(dtype=torch.float)
                loss_fct = BCEWithLogitsLoss(pos_weight=weight, reduction="none")  # TODO: reduction?
                # loss = loss_fct(logits.view(-1, len(self.labels)), labels.view(-1, len(self.labels)))
                loss = loss_fct(logits, labels)


        return loss
