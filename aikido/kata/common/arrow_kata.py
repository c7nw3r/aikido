import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Union, List, Dict, Optional, Any, Callable

from datasets import Dataset

from aikido.__api__.kata import Kata
from aikido.__api__.kata_spec import KataSpec

logger = logging.getLogger(__name__)


class Preprocessor(ABC):
    """
    A preprocessor is used to transform the raw data before we can feed them to our model. For example a preprocessor
    can tokenize the inputs, put it in a format the model expects and generate the other inputs that model requires.
    """

    @abstractmethod
    def preprocess(self, exercise: Dict, index: Optional[int]) -> Union[Dict, Any]:
        pass

    @abstractmethod
    def preprocess_batch(self, exercises: Dict[str, List], index: Optional[List[int]]) -> Union[Dict, Any]:
        pass


@dataclass
class ArrowKata(Kata):
    spec: KataSpec
    preprocessor: Union[Preprocessor, Callable]

    @abstractmethod
    def get_dataset(self) -> Dataset:
        pass

    def preprocess(self, dataset: Dataset) -> Dataset:
        if not callable(self.preprocessor):
            function = self.preprocessor.preprocess_batch \
                if self.spec.batch_preprocess \
                else self.preprocessor.preprocess
        else:
            function = self.preprocessor

        return dataset.map(function=function,
                           batched=self.spec.batch_preprocess,
                           batch_size=self.spec.batch_preprocess_size,
                           remove_columns=self.spec.remove_columns)
