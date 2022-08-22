import numpy as np
from unittest import TestCase

from PIL import Image

from aikido.kata.v2.triplet_dataset import TripletDataset, AugmentationAwareTripletKata
from aikido.visual.triplet import visualize


class TripletDatasetTest(TestCase):

    def test_abc(self):
        def transform(image: Image.Image):
            image = image.resize((500, 500), Image.BILINEAR)
            array = np.asarray(image)
            array = array / 255
            # array = array.transpose((2, 0, 1))
            array = np.expand_dims(array, axis=0).astype("float32")
            return array

        dataset = TripletDataset("/home/christian/Pictures/topshot/topshot-dataset", transform)
        # for (a, p, n), _ in dataset:
        #     print(a, p, n)

        (a, p, n), _ = dataset[0]
        visualize(a, p, n)

    def test_get_labels(self):
        def transform(image: Image.Image):
            image = image.resize((500, 500), Image.BILINEAR)
            array = np.asarray(image)
            array = array / 255
            # array = array.transpose((2, 0, 1))
            array = np.expand_dims(array, axis=0).astype("float32")
            return array

        dataset = TripletDataset("/home/christian/Pictures/topshot/topshot-dataset", transform)
        print(dataset.get_labels())

    def test_augmentation(self):
        def transform(image: Image.Image):
            image = image.resize((500, 500), Image.BILINEAR)
            array = np.asarray(image)
            array = array / 255
            # array = array.transpose((2, 0, 1))
            array = np.expand_dims(array, axis=0).astype("float32")
            return array

        dataset = AugmentationAwareTripletKata("/home/christian/Pictures/topshot/topshot-dataset", transform)
        # for (a, p, n), _ in dataset:
        #     print(a, p, n)

        # output = dataset[1]
        # visualize(output["anchor"], output["positive"], output["negative"])

        for entry in dataset:
            print(entry["anchor"].shape, entry["positive"].shape)