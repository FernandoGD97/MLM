"""Checkpoint-specific masked predictive entropy, computed outside the collator."""

from __future__ import annotations

import time
from collections.abc import Sequence
from typing import Any

import torch

from lab.cpt.contracts import Window
from lab.cpt.modeling import bidirectional_attention


def predictive_entropy(
    *, model: Any, tokenizer: Any, windows: Sequence[Window]
) -> tuple[dict[tuple[str, int], tuple[float, ...]], float]:
    """Mask each word jointly and average its subword predictive entropies.

    The caller owns refresh frequency and must invalidate this cache for another model or
    corpus. Return measured inference seconds, separately from training throughput.
    """
    started = time.perf_counter()
    result = {}
    device = next(model.parameters()).device
    with bidirectional_attention(model=model), torch.inference_mode():
        model.eval()
        for window in windows:
            scores = []
            for word in window.words:
                ids = torch.tensor([window.input_ids], device=device)
                ids[0, list(word.positions)] = tokenizer.mask_token_id
                logits = (
                    model(input_ids=ids, attention_mask=torch.ones_like(ids))
                    .logits[0, list(word.positions)]
                    .float()
                )
                log_probabilities = logits.log_softmax(-1)
                scores.append(float(-(log_probabilities.exp() * log_probabilities).sum(-1).mean()))
            result[(window.doc_id, window.index)] = tuple(scores)

    return result, time.perf_counter() - started
