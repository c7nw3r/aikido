import os
from collections import OrderedDict
from glob import glob
from os.path import join
from pathlib import Path
from typing import List

import numpy as np
from PIL import Image

# noinspection DuplicatedCode
from aikido.__api__.kata import Kata, LabelAware
from aikido.__util__.images import load_image
from aikido.__util__.lru_cache import LRUCache


class TripletDataset(Kata, LabelAware):
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

        dir_a = Path(self.a_images[index]).parent.as_posix()
        dir_n = np.random.choice(list(set(self.path_map.values()) - {dir_a}))
        img_n = self.random_state.choice(os.listdir(dir_n))

        img1 = self.preprocess(self.a_images[index])
        img2 = self.preprocess(self.p_images[index])
        img3 = self.preprocess(join(dir_n, str(img_n, "utf-8")))

        return OrderedDict({
            "anchor": img1,
            "positive": img2,
            "negative": img3
        })

    def __len__(self):
        return len(self.a_images)

    def get_labels(self) -> List[int]:
        return [i for i in self.path_map.keys() for _ in range(len(os.listdir(self.path_map[i])))]


class AugmentationAwareTripletKata(TripletDataset):

    def __init__(self, folder_path: str, preprocessor, seed: int = None, cache: bool = False):
        super().__init__(folder_path, preprocessor, seed, cache)

    def __getitem__(self, index):
        index = int(index / 2)

        if index % 2 == 0:
            return super().__getitem__(index)

        output = super().__getitem__(index - 1)

        # FIXME: mandatory dependency
        import imgaug.augmenters as iaa

        seq = iaa.Sequential([
            iaa.Crop(px=(1, 16), keep_size=True),
            iaa.Fliplr(1),
            iaa.GaussianBlur(sigma=(0, 3.0))
        ])

        w = output["anchor"].shape[1]
        h = output["anchor"].shape[2]

        anchor_path = self.a_images[index - 1]
        with Image.open(anchor_path) as img:
            img = img.resize((w, h), Image.BILINEAR)
            positive = seq(images=[np.array(img)])[0]
            positive = self.preprocessor(Image.fromarray(positive))

            return OrderedDict({
                "anchor": output["anchor"],
                "positive": positive,
                "negative": output["negative"]
            })

    def __len__(self):
        return len(self.a_images) * 2

    def get_labels(self) -> List[int]:
        return [i for i in self.path_map.keys() for _ in range(len(os.listdir(self.path_map[i]))) for _ in range(2)]

