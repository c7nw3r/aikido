from torch.nn import Module
from transformers import PreTrainedModel, AutoModelPreTrainedModel


class LanguageModel:

    @classmethod
    def load(cls, model_ref, **kwargs):
        model = AutoModel.from_pretrained(model_ref, **kwargs)

        if model.config.model_type == "bert":
            return BertWrapper(model)
        if model.config.model_type == "electra":
            return ElectraWrapper(model)
        raise ValueError(f"cannot load model type {model.config.model_type}")

    @classmethod
    def _infer_language_from_name(cls, name) -> str:
        known_languages = ("german", "english", "chinese", "indian", "french", "polish", "spanish", "multilingual")
        matches = [lang for lang in known_languages if lang in name]
        if "camembert" in name:
            return "french"
        elif "umberto" in name:
            return "italian"
        return matches[0] if len(matches) > 0 else "english"


class BertWrapper(Module):
    def __init__(self, model: PreTrainedModel):
        super().__init__()
        self.model = model

    def forward(self, **kwargs):
        return self.model(input_ids=kwargs.get("input_ids"),
                          attention_mask=kwargs.get("attention_mask"),
                          token_type_ids=kwargs.get("token_type_ids"),
                          position_ids=kwargs.get("position_ids"),
                          head_mask=kwargs.get("head_mask"),
                          inputs_embeds=kwargs.get("inputs_embeds"),
                          encoder_hidden_states=kwargs.get("encoder_hidden_states"),
                          encoder_attention_mask=kwargs.get("encoder_attention_mask"),
                          past_key_values=kwargs.get("past_key_values"),
                          use_cache=kwargs.get("use_cache"),
                          output_attentions=kwargs.get("output_attentions"),
                          output_hidden_states=kwargs.get("output_hidden_states"),
                          return_dict=kwargs.get("return_dict"))

    @property
    def config(self):
        return self.model.config

class ElectraWrapper(Module):
    def __init__(self, model: PreTrainedModel):
        super().__init__()
        self.model = model

    def forward(self, **kwargs):
        return self.model(input_ids=kwargs.get("input_ids"),
                          attention_mask=kwargs.get("attention_mask"),
                          token_type_ids=kwargs.get("token_type_ids"),
                          position_ids=kwargs.get("position_ids"),
                          head_mask=kwargs.get("head_mask"),
                          inputs_embeds=kwargs.get("inputs_embeds"),
                          output_attentions=kwargs.get("output_attentions"),
                          output_hidden_states=kwargs.get("output_hidden_states"),
                          return_dict=kwargs.get("return_dict"))

    @property
    def config(self):
        return self.model.config