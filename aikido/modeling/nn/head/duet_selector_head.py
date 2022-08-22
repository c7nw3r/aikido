from aikido.__api__.aikidoka import Aikidoka
from aikido.__util__.tensors import unsqueeze4d, to_tensor
from aikido.__util__.value_dict import ValueDict
from aikido.modeling.loss.duet.duet_selector import PairSelector
from aikido.modeling.loss.online_contrastive_loss import OnlineContrastiveLoss
from aikido.modeling.loss.online_triplet_loss import OnlineTripletLoss
from aikido.modeling.loss.triplet.triplet_selector import TripletSelector


class DuetSelectorHead(Aikidoka):
    def __init__(self, embedding_net: Aikidoka, duet_selector: PairSelector):
        super(DuetSelectorHead, self).__init__()
        self.embedding_net = embedding_net
        self.loss_fn = OnlineContrastiveLoss(1., duet_selector)

    def forward(self, x, y):
        x = self.embedding_net(x)
        loss = self.loss_fn.forward(x, y)
        return ValueDict({"loss": loss})

    def get_embedding(self, x):
        return self.embedding_net(unsqueeze4d(to_tensor(x)))
