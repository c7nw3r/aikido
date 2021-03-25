import logging
import os
from dataclasses import dataclass
from typing import Union

from tokenizers import Tokenizer
from transformers import AutoTokenizer, DataCollatorWithPadding

from aikido.__api__.kata_spec import KataSpec
from aikido.__api__.trait.file_trait import _download_extract_downstream_data
from aikido.kata.tsv_kata import TsvKata

logger = logging.getLogger(__name__)


@dataclass
class GermEval18:
    folder: str
    tokenizer: Union[str, Tokenizer]
    spec: KataSpec = KataSpec()

    def __post_init__(self):
        if type(self.tokenizer) is str:
            self.tokenizer = AutoTokenizer.from_pretrained(self.tokenizer, use_fast=True)

    # noinspection PyTypeChecker
    def katas(self):
        if not (os.path.exists(self.folder)):
            logger.info(f"Download GermEval18 dataset")
            _download_extract_downstream_data(self.folder + "/train.tsv", proxies=None)

        def preprocess_function(examples):
            labels = GermEval18.labels(True)
            examples["coarse_label"] = [labels.index(x) for x in examples["coarse_label"]]

            labels = GermEval18.labels(False)
            examples["fine_label"] = [labels.index(x) for x in examples["fine_label"]]

            assert callable(self.tokenizer), "tokenizer is not callable"
            return self.tokenizer(examples["text"], truncation=True,
                                  return_token_type_ids=True,  # FIXME
                                  max_length=self.spec.max_seq_length
                                  )

        spec = KataSpec(remove_columns=["text"], max_seq_length=self.spec.max_seq_length)
        collator = DataCollatorWithPadding(self.tokenizer, padding="max_length", max_length=self.spec.max_seq_length)
        return (
            TsvKata(spec, preprocess_function, collator, "../data/germeval18/train.tsv"),
            TsvKata(spec, preprocess_function, collator, "../data/germeval18/test.tsv")
        )

    @staticmethod
    def labels(coarse: bool):
        return ["OTHER", "OFFENSE"] if coarse else ["OTHER", "OFFENSE", "ABUSE", "INSULT", "PROFANITY"]
