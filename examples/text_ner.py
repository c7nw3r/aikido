import logging

from aikido.__api__.dojo_kun import DojoKun
from aikido.aikidoka.adaptive_transformer import AdaptiveTransformer
from aikido.dojo.base_dojo import BaseDojo
from aikido.dojo.listener import CudaListener, LearningRateStepListener, SeedListener, EvaluationListener, \
    ValidationListener
from aikido.dojo.listener.gradient_clipping_listener import GradientClippingListener
from aikido.kata.factory.conll03 import Conll03
from aikido.kata.factory.germeval14 import GermEval14
from aikido.modeling.language_model import LanguageModel
from aikido.modeling.nn.head.token_classifier_head import TokenClassifierHead
from aikido.modeling.optimization import transformers_adam_w_optimizer

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    model_name = "bert-base-german-cased"
    train, val, test = Conll03(model_name, lang="de")()
    # train, val, test = GermEval14(model_name)()

    language_model = LanguageModel.load(model_name, add_pooling_layer=False)
    prediction_head = TokenClassifierHead(labels=GermEval14.labels())
    model = AdaptiveTransformer(language_model, [prediction_head])

    optimizer = transformers_adam_w_optimizer(learning_rate=2e-5)

    listeners = [
        CudaListener(reset_memory=True),
        LearningRateStepListener(),
        GradientClippingListener(),
        ValidationListener(test, "seq_f1"),
        EvaluationListener(val, metrics=Conll03.metrics(), evaluate_every=200),
        SeedListener(40)
    ]

    dojo = BaseDojo(DojoKun(optimizer, dans=3, batch_size=24), listeners)
    dojo.train(model, train)

    samples = [
        {"text": "Schartau sagte dem Tagesspiegel, dass Fischer ein Idiot sei"},
        {"text": "Martin Müller spielt Handball in Berlin"},
    ]
