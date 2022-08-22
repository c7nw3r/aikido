import numpy as np
from PIL import Image

from aikido.__api__.kata import Preprocessor


class ResNetPreprocessor(Preprocessor[Image.Image]):

    def __init__(self):
        # noinspection PyArgumentList
        self.mean = np.array([0.485, 0.456, 0.406]).reshape(-1, 1, 1)
        # noinspection PyArgumentList
        self.std = np.array([0.229, 0.224, 0.225]).reshape(-1, 1, 1)

    def __call__(self, image: Image.Image):
        # noinspection PyUnresolvedReferences
        image = image.resize((224, 224), Image.Resampling.BILINEAR)
        # noinspection PyTypeChecker
        array = np.asarray(image)
        array = array / 255
        array = array.transpose((2, 0, 1))
        array = array - self.mean
        array = array / self.std
        return array.astype("float32")
