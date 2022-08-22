import logging
from abc import abstractmethod, ABC
from typing import TypeVar, Generic, List, Sized

from torch.utils.data import random_split, Dataset
# from transformers import default_data_collator, DataCollator
from torch.utils.data.dataset import T_co, Subset

logger = logging.getLogger(__name__)


class Kata(Dataset):

    # def load(self, batch_size: int = 1) -> LoadedKata:
    #     dataset = self.get_dataset()
    #     sampler = DistributedSampler(dataset) if self.spec.distributed else RandomSampler(dataset)
    #
    #     return LoadedKata(str(uuid.uuid4()), DataLoader(
    #         dataset=dataset,
    #         batch_size=batch_size,
    #         sampler=sampler,
    #         collate_fn=self.get_data_collator(),
    #         drop_last=self.spec.dataloader_drop_last,
    #         num_workers=self.spec.dataloader_num_workers
    #     ))

    def split(self, ratio: float) -> ('Kata', 'Kata'):
        assert 0 < ratio < 1, "ratio must be between 0 and 1"

        size1 = int(ratio * len(self))
        size2 = len(self) - size1
        subset1, subset2 = random_split(self, [size1, size2])
        return SubsetKata(subset1), SubsetKata(subset2)

    @abstractmethod
    def __len__(self):
        pass


class LabelAware(Sized):

    @abstractmethod
    def get_labels(self) -> List[int]:
        pass


class SubsetKata(Kata, LabelAware):

    def __init__(self, subset: Subset):
        self.subset = subset

    def __len__(self):
        return len(self.subset)

    def __getitem__(self, index) -> T_co:
        return self.subset[index]

    def get_labels(self) -> List[int]:
        kata = self.subset.dataset
        if issubclass(type(kata), LabelAware):
            # noinspection PyUnresolvedReferences
            labels = kata.get_labels()
            return [labels[i] for i in self.subset.indices]
        raise NotImplementedError()


T = TypeVar("T")


class Preprocessor(Generic[T]):

    def __call__(self, value: T):
        pass
