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
class CsvKata(ArrowKata):
    """ Kata implementation which loads a tsv file. """
    spec: KataSpec
    preprocessor: Union[Preprocessor, Callable]
    collator: DataCollator
    file_path: str
    quote_char: str = "'"
    skiprows: int = 0
    column_names: Optional[List[str]] = None
    delimiter: str = ','
    encoding: str = 'utf-8'

    def get_dataset(self) -> Dataset:
        # https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.read_csv.html
        return self.preprocess(load_dataset('csv',
                                            data_files=[self.file_path],
                                            skiprows=self.skiprows,
                                            column_names=self.column_names,
                                            delimiter=self.delimiter,
                                            quotechar=self.quote_char,
                                            encoding=self.encoding)["train"])

    def get_data_collator(self) -> DataCollator:
        return self.collator
