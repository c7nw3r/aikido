import logging
from dataclasses import dataclass
from typing import Union, Callable, List, Optional

from datasets import load_dataset
from torch.utils.data import Dataset
from transformers import DataCollator

from aikido.__api__.kata_spec import KataSpec
from aikido.kata.common.arrow_kata import ArrowKata, Preprocessor

logger = logging.getLogger(__name__)


@dataclass
class JsonKata(ArrowKata):
    """ Kata implementation which loads a json file. """
    spec: KataSpec
    preprocessor: Union[Preprocessor, Callable]
    collator: DataCollator
    data_files: List[str]
    field: Optional[str] = None

    def get_dataset(self) -> Dataset:
        return self.preprocess(load_dataset('json',
                                            data_files=[self.data_files],
                                            field=self.field,
                                            encoding="utf-8")["train"])

    def get_data_collator(self) -> DataCollator:
        return self.collator
