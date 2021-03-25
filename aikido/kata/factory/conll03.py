from dataclasses import dataclass
from typing import Union

import numpy as np
from datasets import load_dataset, load_metric
from tokenizers import Tokenizer
from transformers import AutoTokenizer, DataCollatorForTokenClassification, PreTrainedTokenizerFast

from aikido.__api__.kata import DatasetKata
from aikido.__api__.kata_spec import KataSpec
from aikido.__api__.metric import MetricRequest, Metric


class Conll2003Metric(Metric):
    def __init__(self):
        self.metric = load_metric("seqeval")

    def __call__(self, request: MetricRequest) -> dict:
        # Remove ignored index (special tokens)
        preds = [
            [Conll03.labels()[p] for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(np.argmax(request.preds, axis=2), request.refs)
        ]
        refs = [
            [Conll03.labels()[l] for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(np.argmax(request.preds, axis=2), request.refs)
        ]

        results = self.metric.compute(predictions=preds, references=refs)
        return {
            "precision": results["overall_precision"],
            "recall": results["overall_recall"],
            "f1": results["overall_f1"],
            "accuracy": results["overall_accuracy"],
        }


@dataclass
class Conll03:
    tokenizer: Union[str, Tokenizer]
    spec: KataSpec = KataSpec()
    lang: str = "en"
    task: str = "ner"
    label_all_tokens: bool = True

    def __post_init__(self):
        if type(self.tokenizer) is str:
            self.tokenizer = AutoTokenizer.from_pretrained(self.tokenizer, use_fast=True)
            assert isinstance(self.tokenizer, PreTrainedTokenizerFast)

    # noinspection PyTypeChecker
    def __call__(self):
        if self.lang == "de":
            # dataset = load_dataset('../config/conll2003-de/', 'conll2003-de', cache_dir=self.spec.cache_dir)
            dataset = load_dataset('D:/IntellijProjects/aikido2/aikido/kata/config/conll2003-de/', 'conll2003-de', cache_dir=self.spec.cache_dir)
        elif self.lang == "en":
            dataset = load_dataset("conll2003", cache_dir=self.spec.cache_dir)
        else:
            raise ValueError(f"cannot handle language {self.lang} for conll2003")

        encoded = dataset.map(function=self.tokenize_and_align_labels,
                              batched=self.spec.batch_preprocess,
                              batch_size=self.spec.batch_preprocess_size,
                              remove_columns=["id", "tokens", "pos_tags", "chunk_tags", "ner_tags"])

        data_collator = DataCollatorForTokenClassification(self.tokenizer)

        return (
            DatasetKata(self.spec, encoded["train"], data_collator),
            DatasetKata(self.spec, encoded["validation"], data_collator),
            DatasetKata(self.spec, encoded["test"], data_collator)
        )

    def tokenize_and_align_labels(self, examples):
        assert callable(self.tokenizer), "tokenizer is not callable"
        tokenized_inputs = self.tokenizer(examples["tokens"], truncation=True, is_split_into_words=True,
                                          return_token_type_ids=False)

        labels = []
        for i, label in enumerate(examples[f"{self.task}_tags"]):
            word_ids = tokenized_inputs.word_ids(batch_index=i)
            previous_word_idx = None
            label_ids = []
            for word_idx in word_ids:
                # Special tokens have a word id that is None. We set the label to -100 so they are automatically
                # ignored in the loss function.
                if word_idx is None:
                    label_ids.append(-100)
                # We set the label for the first token of each word.
                elif word_idx != previous_word_idx:
                    label_ids.append(label[word_idx])
                # For the other tokens in a word, we set the label to either the current label or -100, depending on
                # the label_all_tokens flag.
                else:
                    label_ids.append(label[word_idx] if self.label_all_tokens else -100)
                previous_word_idx = word_idx

            labels.append(label_ids)

        max = np.max(list(map(lambda x: len(x), tokenized_inputs["input_ids"])))

        tokenized_inputs["labels"] = labels
        return tokenized_inputs

    @staticmethod
    def labels():
        return ["[PAD]", "X", "O", "B-MISC", "I-MISC", "B-PER", "I-PER", "B-ORG", "I-ORG", "B-LOC", "I-LOC", "B-OTH",
                "I-OTH"]

    @staticmethod
    def metrics():
        return [Conll2003Metric()]
