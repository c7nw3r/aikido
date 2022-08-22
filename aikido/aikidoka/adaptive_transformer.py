from typing import List

from torch.utils.data import DataLoader

from aikido.__api__.aikidoka import Aikidoka
from aikido.modeling.nn.head import PredictionHead


class AdaptiveTransformer(Aikidoka):
    """ PyTorch implementation containing all the modelling needed for your NLP task. Combines a language
    model and a prediction head. Allows for gradient flow back to the language model component."""
    def __init__(self, language_model: Aikidoka, prediction_heads: List[PredictionHead]):
        super().__init__()
        self.language_model = language_model
        self.prediction_heads = prediction_heads
        for prediction_head in prediction_heads:
            prediction_head.init(language_model.config)

    def forward(self, **batch):
        outputs = self.language_model(**batch, return_dict=True)
        return [head(outputs, batch) for head in self.prediction_heads]

    def to(self, *args, **kwargs):
        super().to(*args, **kwargs)
        for prediction_head in self.prediction_heads:
            prediction_head.to(*args, **kwargs)

    def init_balance(self, data: DataLoader):
        for prediction_head in self.prediction_heads:
            if hasattr(prediction_head, "init_balance"):
                assert callable(prediction_head.init_balance)
                prediction_head.init_balance(data)
        # TODO: maybe return BalancedAdaptiveTransformer class