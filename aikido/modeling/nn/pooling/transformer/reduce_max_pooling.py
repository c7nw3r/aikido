from dataclasses import dataclass

import numpy as np
from transformers.modeling_outputs import BaseModelOutputWithPoolingAndCrossAttentions

from aikido.modeling.nn.pooling.transformer.abstract_transformer_pooling import AbstractTransformerPooling


@dataclass
class ReduceMeanPooling(AbstractTransformerPooling):
    ignore_first_token: bool = False

    def forward(self, x: BaseModelOutputWithPoolingAndCrossAttentions):
        token_vecs = x.sequence_output.cpu().numpy()
        # we only take the aggregated value of non-padding tokens
        padding_mask = x.padding_mask.cpu().numpy()
        ignore_mask_2d = padding_mask == 0
        # sometimes we want to exclude the CLS token as well from our aggregation operation
        if self.ignore_first_token:
            ignore_mask_2d[:, 0] = True
        ignore_mask_3d = np.zeros(token_vecs.shape, dtype=bool)
        ignore_mask_3d[:, :, :] = ignore_mask_2d[:, :, np.newaxis]

        return np.ma.array(data=token_vecs, mask=ignore_mask_3d).max(axis=1).data
