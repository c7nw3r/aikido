from typing import List, Optional

import numpy as np
from torch.utils.data.sampler import Sampler

from aikido.__api__.kata import LabelAware


class BalancedBatchSampler(Sampler[List[int]]):
    """
    tbd
    """

    def __init__(self, kata: LabelAware, n_classes, n_samples, seed: Optional[int] = None):
        super().__init__(kata)
        self.random_state = np.random.RandomState(seed)
        self.labels = kata.get_labels()
        self.labels_set = list(set(np.array(self.labels)))
        self.label_to_indices = {label: np.where(np.array(self.labels) == label)[0]
                                 for label in self.labels_set}
        for l in self.labels_set:
            self.random_state.shuffle(self.label_to_indices[l])
        self.used_label_indices_count = {label: 0 for label in self.labels_set}
        self.count = 0
        self.n_classes = n_classes
        self.n_samples = n_samples
        self.n_dataset = len(self.labels)
        self.batch_size = self.n_samples * self.n_classes

    def __iter__(self):
        self.count = 0
        while self.count + self.batch_size < self.n_dataset:
            classes = self.random_state.choice(self.labels_set, self.n_classes, replace=False)
            indices = []
            for class_ in classes:
                indices.extend(self.label_to_indices[class_][
                               self.used_label_indices_count[class_]:self.used_label_indices_count[
                                                                         class_] + self.n_samples])
                self.used_label_indices_count[class_] += self.n_samples
                if self.used_label_indices_count[class_] + self.n_samples > len(self.label_to_indices[class_]):
                    np.random.shuffle(self.label_to_indices[class_])
                    self.used_label_indices_count[class_] = 0

            while len(indices) < self.batch_size:
                indices.append(self.random_state.choice(indices))

            yield indices
            self.count += self.n_classes * self.n_samples

    def __len__(self):
        return self.n_dataset // self.batch_size
