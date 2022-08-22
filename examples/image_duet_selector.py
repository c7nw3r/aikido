from torch.utils.data import DataLoader

from aikido.__api__.dojo_kun import DojoKun
from aikido.__api__.kata import Kata
from aikido.__util__.image_preprocessors import ResNetPreprocessor
from aikido.aikidoka.impl.resnet import ResNet
from aikido.dojo import BaseDojo
from aikido.dojo.listener import EvaluationListener
from aikido.dojo.listener.seed_listener import set_all_seeds
from aikido.kata.image_kata import ImageKata
from aikido.kata.sampler.balanced_batch_sampler import BalancedBatchSampler
from aikido.modeling.loss.duet.duet_selector import HardNegativePairSelector
from aikido.modeling.loss.triplet.triplet_selector import HardestNegativeTripletSelector
from aikido.modeling.nn.head.duet_selector_head import DuetSelectorHead
from aikido.modeling.nn.head.triplet_selector_head import TripletSelectorHead

set_all_seeds(123)


class DuetDojo(BaseDojo):

    def load_kata(self, kata: Kata, cuda: bool) -> DataLoader:
        # noinspection PyTypeChecker
        sampler = BalancedBatchSampler(kata, 5, 2)
        return DataLoader(kata, batch_sampler=sampler,
                          num_workers=1 if cuda else None, pin_memory=cuda)


dataset_train = ImageKata("/home/christian/Pictures/topshot/topshot-dataset", ResNetPreprocessor())
dataset_train, dataset_test = dataset_train.split(0.8)

aikidoka = DuetSelectorHead(ResNet(headless=True), HardNegativePairSelector())

dojo = DuetDojo(DojoKun(dans=10, batch_size=24), EvaluationListener(dataset_test, [], evaluate_every=5))
dojo.train(aikidoka, dataset_train)

aikidoka.save("./model2-duet.pt")
