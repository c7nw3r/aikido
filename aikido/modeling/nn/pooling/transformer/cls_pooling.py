from transformers.modeling_outputs import BaseModelOutputWithPoolingAndCrossAttentions

from aikido.modeling.nn.pooling.transformer.abstract_transformer_pooling import AbstractTransformerPooling


class ClsPooling(AbstractTransformerPooling):

    def forward(self, x: BaseModelOutputWithPoolingAndCrossAttentions):
        return x[0][:, 0, :].detach().cpu().numpy()