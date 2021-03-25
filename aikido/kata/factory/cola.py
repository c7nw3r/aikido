from dataclasses import dataclass
from typing import Union

from datasets import load_dataset
from tokenizers import Tokenizer
from transformers import AutoTokenizer, DataCollatorWithPadding

from aikido.__api__.kata import DatasetKata
from aikido.__api__.kata_spec import KataSpec


@dataclass
class Cola:
    tokenizer: Union[str, Tokenizer]
    spec: KataSpec = KataSpec()

    def __post_init__(self):
        if type(self.tokenizer) is str:
            self.tokenizer = AutoTokenizer.from_pretrained(self.tokenizer, use_fast=True)

    # noinspection PyTypeChecker
    def __call__(self):
        dataset = load_dataset(path="glue", name="cola", cache_dir=self.spec.cache_dir)
        encoded = dataset.map(function=self.preprocess_function,
                              batched=self.spec.batch_preprocess,
                              batch_size=self.spec.batch_preprocess_size,
                              remove_columns=["sentence"])

        data_collator = DataCollatorWithPadding(self.tokenizer, "max_length", self.spec.max_seq_length)

        return (
            DatasetKata(self.spec, encoded["train"], data_collator),
            DatasetKata(self.spec, encoded["validation"], data_collator),
            DatasetKata(self.spec, encoded["test"], data_collator)
        )

    def preprocess_function(self, examples):
        assert callable(self.tokenizer), "tokenizer is not callable"
        return self.tokenizer(examples["sentence"], truncation=True,
                              return_token_type_ids=True  # FIXME
                              )

    @staticmethod
    def labels():
        return ["0", "1"]
