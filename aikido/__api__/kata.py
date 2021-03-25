import logging
import uuid
from abc import abstractmethod
from dataclasses import dataclass
from typing import Optional

from torch.utils.data import DistributedSampler, RandomSampler, DataLoader, random_split, Dataset
from transformers import default_data_collator, DataCollator

from aikido.__api__.kata_spec import KataSpec

logger = logging.getLogger(__name__)


@dataclass
class LoadedKata:
    uuid: str
    data_loader: DataLoader

    def __len__(self):
        # noinspection PyTypeChecker
        return len(self.data_loader.dataset)


@dataclass
class Kata:
    spec: KataSpec

    def load(self, batch_size: int = 1) -> LoadedKata:
        dataset = self.get_dataset()
        sampler = DistributedSampler(dataset) if self.spec.distributed else RandomSampler(dataset)

        return LoadedKata(str(uuid.uuid4()), DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            sampler=sampler,
            collate_fn=self.get_data_collator(),
            drop_last=self.spec.dataloader_drop_last,
            num_workers=self.spec.dataloader_num_workers
        ))

    # noinspection PyTypeChecker
    def get_data_collator(self) -> DataCollator:
        return default_data_collator

    @abstractmethod
    def get_dataset(self):
        pass

    def split(self, ratio: float) -> ('Kata', 'Kata'):
        assert 0 < ratio < 1, "ratio must be between 0 and 1"

        size1 = int(ratio * len(self.get_dataset()))
        size2 = len(self.get_dataset()) - size1
        collator = self.get_data_collator()
        dataset1, dataset2 = random_split(self.get_dataset(), [size1, size2])
        return DatasetKata(self.spec, dataset1, collator), DatasetKata(self.spec, dataset2, collator)


@dataclass
class DatasetKata(Kata):
    spec: KataSpec
    dataset: Dataset
    data_collator: Optional[DataCollator] = None

    def get_dataset(self) -> Dataset:
        return self.dataset

    def get_data_collator(self) -> DataCollator:
        if self.data_collator:
            return self.data_collator
        return super().get_data_collator()
