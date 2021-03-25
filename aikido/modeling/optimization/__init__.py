import logging
import sys
from importlib import import_module

# Used indirectly in _get_optim() to avoid name collision with torch's AdamW
from aikido.__api__.aikidoka import Aikidoka

logger = logging.getLogger(__name__)


def transformers_adam_w_optimizer(correct_bias: bool = False, weight_decay: float = 0.01, learning_rate: float = 3e-5):
    return _get_optim({
        "name": "AdamW", # FIXME use optimizer from transformers package
        "correct_bias": correct_bias,
        "weight_decay": weight_decay,
        "lr": learning_rate
    })


def _get_optim(opts):
    """ Get the optimizer based on dictionary with options. Options are passed to the optimizer constructor.

    :param model: model to optimize
    :param opts: config dictionary that will be passed to optimizer together with the params
    (e.g. lr, weight_decay, correct_bias ...). no_decay' can be given - parameters containing any of those strings
    will have weight_decay set to 0.
    :return: created optimizer
    """

    def get_optimizer(model: Aikidoka):
        optimizer_name = opts.pop('name', None)

        logger.info(f"Loading optimizer `{optimizer_name}`: '{opts}'")

        weight_decay = opts.pop('weight_decay', None)
        no_decay = opts.pop('no_decay', None)

        if no_decay:
            optimizable_parameters = [
                {'params': [p for n, p in model.named_parameters() if
                            not any(nd in n for nd in no_decay) and p.requires_grad],
                 **opts},
                {'params': [p for n, p in model.named_parameters() if
                            any(nd in n for nd in no_decay) and p.requires_grad],
                 'weight_decay': 0.0,
                 **opts}
            ]
        else:
            optimizable_parameters = [{'params': [p for p in model.parameters() if p.requires_grad], **opts}]

        # default weight decay is not the same for all optimizers, so we can't use default value
        # only explicitly add weight decay if it's given
        if weight_decay is not None:
            optimizable_parameters[0]['weight_decay'] = weight_decay

        # Import optimizer by checking in order: torch, transformers, apex and local imports
        try:
            optim_constructor = getattr(import_module('torch.optim'), optimizer_name)
        except AttributeError:
            try:
                optim_constructor = getattr(import_module('transformers.optimization'), optimizer_name)
            except AttributeError:
                try:
                    optim_constructor = getattr(import_module('apex.optimizers'), optimizer_name)
                except (AttributeError, ImportError):
                    try:
                        # Workaround to allow loading AdamW from transformers
                        # pytorch > 1.2 has now also a AdamW (but without the option to set bias_correction = False,
                        # which is done in the original BERT implementation)
                        optim_constructor = getattr(sys.modules[__name__], optimizer_name)
                    except (AttributeError, ImportError):
                        raise AttributeError(
                            f"Optimizer '{optimizer_name}' not found in 'torch', 'transformers', 'apex' or 'local imports")

        return optim_constructor(optimizable_parameters)

    return get_optimizer
