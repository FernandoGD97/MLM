"""Exception-safe, non-persistent attention-mode switches for encoder diagnostics."""
from contextlib import contextmanager
from ..objectives import set_attention_mode

@contextmanager
def bidirectional_evaluation(model):
    was_training = model.training
    previous = bool(getattr(model.config, "is_decoder", False))
    try:
        set_attention_mode(model, "mlm")
        model.eval()
        yield model
    finally:
        set_attention_mode(model, "clm" if previous else "mlm")
        model.train(was_training)
