import logging

from aikido.__api__.dojo_kun import DojoKun
from aikido.__api__.kata_spec import KataSpec
from aikido.aikidoka.adaptive_transformer import AdaptiveTransformer
from aikido.dojo.base_dojo import BaseDojo
from aikido.dojo.listener import LearningRateStepListener, SeedListener
from aikido.dojo.listener.gradient_clipping_listener import GradientClippingListener
from aikido.kata.factory.toxic_comments import ToxicComments
from aikido.modeling.language_model import LanguageModel
from aikido.modeling.nn.head.text_classifier_head import TextClassifierHead
from aikido.modeling.optimization import transformers_adam_w_optimizer

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # model_name = "bert-base-uncased"
    model_name = "german-nlp-group/electra-base-german-uncased"
    kata1, kata2 = ToxicComments("../data/toxic-comments", model_name, KataSpec(max_seq_length=100))()

    language_model = LanguageModel.load(model_name)
    prediction_head = TextClassifierHead(ToxicComments.labels(), "labels", multiclass=True)

    model = AdaptiveTransformer(language_model, [prediction_head])
    # model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=len(ToxicComments.labels()))
    optimizer = transformers_adam_w_optimizer()

    listeners = [LearningRateStepListener(),
                 # AdjustLossListener(),
                 GradientClippingListener(),
                 # EvaluationListener(kata2, metrics="acc", evaluate_every=50),
                 SeedListener(42)]

    dojo = BaseDojo(DojoKun(optimizer, dans=1, batch_size=32, grad_acc_steps=5), listeners)
    dojo.train(model, kata1)

    evaluations = dojo.evaluate(model, kata2)
