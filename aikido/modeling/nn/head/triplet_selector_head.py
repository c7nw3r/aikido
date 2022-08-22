from aikido.__api__.aikidoka import Aikidoka
from aikido.__util__.tensors import unsqueeze4d, to_tensor
from aikido.__util__.value_dict import ValueDict
from aikido.modeling.loss.online_triplet_loss import OnlineTripletLoss
from aikido.modeling.loss.triplet.triplet_selector import TripletSelector


class TripletSelectorHead(Aikidoka):
    def __init__(self, embedding_net, triplet_selector: TripletSelector):
        super(TripletSelectorHead, self).__init__()
        self.embedding_net = embedding_net
        self.loss_fn = OnlineTripletLoss(1., triplet_selector)

    def forward(self, x, y):
        x = self.embedding_net(x)
        loss, _ = self.loss_fn.forward(x, y)
        return ValueDict({"loss": loss})

    def get_embedding(self, x):
        return self.embedding_net(unsqueeze4d(to_tensor(x)))
