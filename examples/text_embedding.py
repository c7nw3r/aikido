from aikido.aikidoka.pooled_transformer import PooledTransformer
from aikido.kata.inference_kata import InferenceKata
from aikido.modeling.language_model import LanguageModel
from aikido.modeling.nn.pooling.transformer.cls_pooling import ClsPooling

if __name__ == "__main__":
    model_name = "bert-base-german-cased"

    language_model = LanguageModel.load(model_name)
    aikidoka = PooledTransformer(language_model, ClsPooling())

    samples = [
        "Schartau sagte dem Tagesspiegel, dass Fischer ein Idiot ist",
        "Martin Müller spielt Fussball"
    ]
    kata = InferenceKata(model_name).of(samples)
    data = kata.load()

    for x in data.data_loader:
        print(aikidoka(**x))
