from dataclasses import dataclass
from typing import Union, Optional, List, overload

from tokenizers import Tokenizer
from transformers import DataCollatorWithPadding, AutoTokenizer

from aikido.__api__.kata_spec import KataSpec
from aikido.kata.dict_kata import DictKata


@dataclass
class InferenceKata:
    tokenizer: Union[str, Tokenizer]
    spec: KataSpec = KataSpec()
    max_length: Optional[int] = None

    def __post_init__(self):
        if type(self.tokenizer) is str:
            self.tokenizer = AutoTokenizer.from_pretrained(self.tokenizer, use_fast=True)

    @overload
    def of(self, values: List[str]):
        ...

    # noinspection PyTypeChecker
    def of(self, *args):
        def preprocess_function(examples):
            assert callable(self.tokenizer), "tokenizer is not callable"
            return self.tokenizer(examples["text"], truncation=True, max_length=self.max_length)

        values = args[0]

        spec = KataSpec(remove_columns=["text"], max_seq_length=self.max_length)
        collator = DataCollatorWithPadding(self.tokenizer, padding="max_length", max_length=self.max_length)
        return DictKata(spec, preprocess_function, collator, {"text": values})