from aikido.__api__.aikidoka import Aikidoka
from aikido.modeling.language_model import LanguageModel
from aikido.modeling.nn.pooling.transformer.abstract_transformer_pooling import AbstractTransformerPooling


class PooledTransformer(Aikidoka):

    def __init__(self, encoder: LanguageModel, pooling: AbstractTransformerPooling):
        super(PooledTransformer, self).__init__()
        self.encoder = encoder
        self.pooling = pooling

    def to(self, *args, **kwargs):
        self.encoder = self.encoder.to(*args, **kwargs)
        self.pooling = self.pooling.to(*args, **kwargs)

    def forward(self, **kwargs):
        return self.pooling(self.encoder(**kwargs))

    def get_language(self):
        return self.language_model.language
