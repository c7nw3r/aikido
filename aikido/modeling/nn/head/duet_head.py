from torch import Tensor

from aikido.__api__.aikidoka import Aikidoka
from aikido.__util__.tensors import to_tensor, unsqueeze4d
from aikido.__util__.value_dict import ValueDict
from aikido.modeling.loss.duet.contrastive_loss import ContrastiveLoss


class DuetHead(Aikidoka):
    def __init__(self, embedding_net, margin: float = 1.):
        super(DuetHead, self).__init__()
        self.embedding_net = embedding_net
        self.loss_fn = ContrastiveLoss(margin)

    def forward(self, image1, image2, label):
        output1 = self.embedding_net(image1)
        output2 = self.embedding_net(image2)

        loss = self.loss_fn.forward(output1, output2, label)
        return ValueDict({"loss": loss})

    def get_embedding(self, x: Tensor):
        return self.embedding_net(unsqueeze4d(to_tensor(x)))
