import logging
import os
from dataclasses import dataclass
from typing import Union, Optional

from tokenizers import Tokenizer
from transformers import AutoTokenizer, DataCollatorWithPadding

from aikido.__api__.kata_spec import KataSpec
from aikido.__api__.trait.file_trait import _download_extract_downstream_data
from aikido.kata.tsv_kata import TsvKata

logger = logging.getLogger(__name__)


@dataclass
class ToxicComments:
    folder: str
    tokenizer: Union[str, Tokenizer]
    spec: KataSpec = KataSpec()

    def __post_init__(self):
        if type(self.tokenizer) is str:
            self.tokenizer = AutoTokenizer.from_pretrained(self.tokenizer, use_fast=True)

    # noinspection PyTypeChecker
    def __call__(self):
        if not (os.path.exists(self.folder)):
            logger.info(f"Download ToxicComments dataset")
            _download_extract_downstream_data(self.folder + "/train.tsv", proxies=None)

        def preprocess_function(examples):
            # TODO: use to utility class
            def hot_encoding(labels: Optional[str]):
                label_ids = [0] * len(ToxicComments.labels())

                if labels is None:
                    return label_ids

                for l in labels.split(","):
                    if l != "":
                        label_ids[ToxicComments.labels().index(l)] = 1
                return label_ids

            examples["label"] = [hot_encoding(x) for x in examples["label"]]

            assert callable(self.tokenizer), "tokenizer is not callable"
            return self.tokenizer(examples["text"], truncation=True,
                                  return_token_type_ids=True,  # FIXME
                                  max_length=self.spec.max_seq_length
                                  )

        spec = KataSpec(remove_columns=["text"], max_seq_length=self.spec.max_seq_length)
        collator = DataCollatorWithPadding(self.tokenizer, padding="max_length", max_length=self.spec.max_seq_length)
        return (
            TsvKata(spec, preprocess_function, collator, "../data/toxic-comments/train.tsv", quote_char='"'),
            TsvKata(spec, preprocess_function, collator, "../data/toxic-comments/val.tsv", quote_char='"')
        )

    @staticmethod
    def labels():
        return ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]