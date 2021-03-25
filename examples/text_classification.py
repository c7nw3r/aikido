import logging

from aikido.__api__.dojo_kun import DojoKun
from aikido.__api__.kata_spec import KataSpec
from aikido.aikidoka.adaptive_transformer import AdaptiveTransformer
from aikido.dojo.base_dojo import BaseDojo
from aikido.dojo.listener import LearningRateStepListener, SeedListener, CudaListener
from aikido.dojo.listener.gradient_clipping_listener import GradientClippingListener
from aikido.kata.factory.germeval18 import GermEval18
from aikido.modeling.language_model import LanguageModel
from aikido.modeling.nn.head.text_classifier_head import TextClassifierHead
from aikido.modeling.optimization import transformers_adam_w_optimizer

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    model_name = "bert-base-german-cased"
    kata1, kata2 = GermEval18("../data/germeval18", model_name, KataSpec(max_seq_length=100)).katas()
    # kata1, kata2 = Cola()()

    language_model = LanguageModel.load(model_name)
    prediction_head = TextClassifierHead(labels=GermEval18.labels(coarse=True), label_name="coarse_label")

    model = AdaptiveTransformer(language_model, [prediction_head])

    optimizer = transformers_adam_w_optimizer()

    listeners = [LearningRateStepListener(),
                 GradientClippingListener(),
                 CudaListener(),
                 # EvaluationListener(kata2, metric, evaluate_every=100),
                 SeedListener(123)]

    dojo = BaseDojo(DojoKun(optimizer, dans=1, batch_size=32), listeners)
    dojo.train(model, kata1)

    evaluations = dojo.evaluate(model, kata2)
