from transformers.modeling_outputs import BaseModelOutputWithPoolingAndCrossAttentions

from aikido.modeling.nn.pooling.transformer.abstract_transformer_pooling import AbstractTransformerPooling


class PooledOutputPooling(AbstractTransformerPooling):

    def forward(self, x: BaseModelOutputWithPoolingAndCrossAttentions):
        assert x.pooled_output is not None, "no pooled output found. maybe extraction layer is not -1"
        return x.pooled_output.detach().cpu().numpy()
