import logging
from dataclasses import dataclass

import torch
from transformers.modeling_outputs import TokenClassifierOutput

from aikido.__api__.aikidoka import AikidokaResult

logger = logging.getLogger(__name__)


@dataclass
class TokenClassifierResult(TokenClassifierOutput, AikidokaResult):

    def as_preds(self):
        preds_tokens = torch.argmax(self.logits, dim=2)
        return preds_tokens.detach().cpu().numpy()

    def as_probs(self, return_class_probs):
        softmax = torch.nn.Softmax(dim=2)
        token_probs = softmax(self.logits)
        if return_class_probs:
            token_probs = token_probs
        else:
            token_probs = torch.max(token_probs, dim=2)[0]
        return token_probs.detach().cpu().numpy()
