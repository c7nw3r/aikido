import logging
from dataclasses import dataclass
from typing import Union, Callable

from datasets import Dataset as HuggingfaceDataset
from torch.utils.data import Dataset
from transformers import DataCollator

from aikido.__api__.kata_spec import KataSpec
from aikido.kata.common.arrow_kata import ArrowKata, Preprocessor

logger = logging.getLogger(__name__)


@dataclass
class DictKata(ArrowKata):
    """ Kata implementation which uses a dictionary. """
    spec: KataSpec
    preprocessor: Union[Preprocessor, Callable]
    collator: DataCollator
    data: dict

    def get_dataset(self) -> Dataset:
        return self.preprocess(HuggingfaceDataset.from_dict(self.data))

    def get_data_collator(self) -> DataCollator:
        return self.collator
