from unittest import TestCase

from aikido.__util__.image_preprocessors import ResNetPreprocessor
from aikido.kata.sampler.balanced_batch_sampler import BalancedBatchSampler
from aikido.kata.v2.triplet_dataset import TripletDataset


class BalancedBatchSamplerTest(TestCase):

    def test_abc(self):
        dataset = TripletDataset("/home/christian/Pictures/topshot/topshot-dataset", ResNetPreprocessor())

        sampler = BalancedBatchSampler(dataset, 5, 3)
        for e in sampler:
            print("---")
            print(e)
