from torch import Tensor

from aikido.__api__.aikidoka import Aikidoka
from aikido.__util__.tensors import to_tensor, unsqueeze4d
from aikido.__util__.value_dict import ValueDict
from aikido.modeling.loss.triplet.triplet_loss import TripletLoss


class TripletNetHead(Aikidoka):
    def __init__(self, embedding_net, margin: float = 1.):
        super(TripletNetHead, self).__init__()
        self.embedding_net = embedding_net
        self.loss_fn = TripletLoss(margin)

    def forward(self, anchor, positive, negative):
        output1 = self.embedding_net(anchor)
        output2 = self.embedding_net(positive)
        output3 = self.embedding_net(negative)

        loss = self.loss_fn.forward(output1, output2, output3)
        return ValueDict({"loss": loss})

    def get_embedding(self, x: Tensor):
        return self.embedding_net(unsqueeze4d(to_tensor(x)))
