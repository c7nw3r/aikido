import os
from collections import OrderedDict
from glob import glob
from typing import List

import numpy as np

# noinspection DuplicatedCode
from aikido.__api__.kata import Kata, LabelAware
from aikido.__util__.images import load_image
from aikido.__util__.lru_cache import LRUCache


class ImageKata(Kata, LabelAware):
    """
    tbd
    """

    def __init__(self, folder_path: str, preprocessor, seed: int = None, cache: bool = True):
        self.images = []
        self.labels = []

        for i, class_ in enumerate(os.listdir(folder_path)):
            img_paths = glob(f'{folder_path}/{class_}/*.jpg')
            self.images.extend(img_paths)
            self.labels.extend([i] * len(img_paths))

        self.preprocessor = preprocessor
        self.random_state = np.random.RandomState(seed)
        self.cache = LRUCache(len(self.images) if cache else 0)

    def preprocess(self, path: str):
        return self.cache.get_or_else(path, lambda x: load_image(x, self.preprocessor))

    def __getitem__(self, index):
        if isinstance(index, list):
            return [self.__getitem__(i) for i in index]

        return OrderedDict({
            "x": self.preprocess(self.images[index]),
            "y": self.labels[index]
        })

    def __len__(self):
        return len(self.images)

    def get_labels(self) -> List[int]:
        return self.labels
