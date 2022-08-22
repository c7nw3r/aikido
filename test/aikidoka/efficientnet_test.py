from unittest import TestCase

from aikido.aikidoka.impl.efficientnet import EfficientNet


class EfficientNetTest(TestCase):

    def test_abc(self):
        model = EfficientNet(headless=True)
        print(model)