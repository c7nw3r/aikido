import os
from glob import glob
from os.path import join
from pathlib import Path
from typing import List

import numpy as np

# noinspection DuplicatedCode
from aikido.__api__.kata import Kata, LabelAware
from aikido.__util__.images import load_image
from aikido.__util__.lru_cache import LRUCache
from aikido.__util__.value_dict import ValueDict


class DuetKata(Kata, LabelAware):
    """
    tbd
    """

    def __init__(self, folder_path: str, preprocessor, seed: int = None, cache: bool = False):
        self.a_images = []  # anchor images
        self.p_images = []  # positive images
        self.path_map = {}  # corresponding paths

        for i, class_ in enumerate(os.listdir(folder_path)):
            img_paths = glob(f'{folder_path}/{class_}/*.jpg')

            if len(img_paths) < 2:
                continue
            if len(img_paths) % 2:
                img_paths = img_paths[1:]

            n = len(img_paths) // 2

            self.a_images.extend(img_paths[:n])
            self.p_images.extend(img_paths[n:])

            self.path_map[i] = join(folder_path, class_)

        self.preprocessor = preprocessor
        self.random_state = np.random.RandomState(seed)
        self.cache = LRUCache(200 if cache else 0)

    def preprocess(self, path: str):
        return self.cache.get_or_else(path, lambda x: load_image(x, self.preprocessor))

    def __getitem__(self, index):
        if isinstance(index, list):
            return [self.__getitem__(i) for i in index]

        label = np.random.randint(0, 2)

        if label == 1:  # positive pair
            img1 = self.preprocess(self.a_images[index])
            img2 = self.preprocess(self.p_images[index])
        else:  # negative pair
            dir_a = Path(self.a_images[index]).parent.as_posix()
            dir_n = np.random.choice(list(set(self.path_map.values()) - {dir_a}))
            img_n = self.random_state.choice(os.listdir(dir_n))

            img1 = self.preprocess(self.a_images[index])
            img2 = self.preprocess(join(dir_n, str(img_n, "utf-8")))

        return {
            "image1": img1,
            "image2": img2,
            "label": label
        }

    def __len__(self):
        return len(self.a_images)

    def get_labels(self) -> List[int]:
        return [i for i in self.path_map.keys() for _ in range(len(os.listdir(self.path_map[i])))]
