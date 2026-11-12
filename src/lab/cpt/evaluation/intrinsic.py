"""Target-weighted fixed-mask CE/accuracy and explicitly conditioned pseudo-likelihood."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any, Literal

import torch
from torch.nn import functional as functional

from lab.cpt.contracts import Span, Window
from lab.cpt.masking.collator import MaskCollator
from lab.cpt.masking.policies import MaskPolicy
from lab.cpt.modeling import bidirectional_attention


def evaluate_pretraining(
    *,
    model: Any,
    tokenizer: Any,
    windows: Sequence[Window],
    objective: str = "mlm",
    probability: float = 0.15,
    evaluation_seed: int = 44,
    batch_size: int = 8,
) -> dict[str, float | int]:
    """Evaluate policy-independent fixed random-subword masks, normalized by targets.

    CLM labels target the next eligible token; exp(CE) is called PPL only in causal mode.
    Zero-target evaluation fails rather than reporting a misleading zero or NaN.
    """
    if objective not in {"mlm", "clm"}:
        raise ValueError("objective must be mlm or clm.")
    if bool(model.config.is_decoder) != (objective == "clm"):
        raise ValueError(
            "Model attention mode differs from evaluation objective; convert explicitly."
        )
    if batch_size < 1:
        raise ValueError("batch_size must be positive.")
    collator = MaskCollator(
        tokenizer=tokenizer,
        policy=MaskPolicy(probability=probability, mask_seed=evaluation_seed),
        objective=objective,
    )
    was_training = model.training
    device = next(model.parameters()).device
    model.eval()
    nll = correct = targets = 0
    try:
        with torch.inference_mode():
            for start in range(0, len(windows), batch_size):
                batch = {
                    key: value.to(device)
                    for key, value in collator(windows[start : start + batch_size]).items()
                }
                labels = batch.pop("labels")
                logits = model(**batch).logits.float()
                valid = labels != -100
                nll += float(
                    functional.cross_entropy(
                        logits.transpose(1, 2), labels, ignore_index=-100, reduction="sum"
                    )
                )
                correct += int(((logits.argmax(-1) == labels) & valid).sum())
                targets += int(valid.sum())
    finally:
        model.train(was_training)
    if not targets:
        raise ValueError("Evaluation has zero targets; increase probability or document length.")
    result = {"nll": nll / targets, "accuracy": correct / targets, "targets": targets}
    if objective == "clm":
        result["causal_ppl"] = math.exp(nll / targets)

    return result


def pseudo_likelihood(
    *,
    model: Any,
    tokenizer: Any,
    windows: Sequence[Window],
    conditioning: Literal["within_word_l2r", "whole_entity"] = "within_word_l2r",
    entities: Mapping[str, tuple[Span, ...]] | None = None,
) -> dict[str, float | int]:
    """Score each token once with a bounded document window and bidirectional context.

    Within-word L2R masks the target and later pieces of its UAX word, retaining earlier
    pieces. Whole-entity PLL masks all tokens of each complete entity simultaneously;
    overlapping entity tokens are scored separately per occurrence, explicitly counted.
    PPPL uses subword-target normalization for both conditionings.
    """
    if conditioning not in {"within_word_l2r", "whole_entity"}:
        raise ValueError("Unknown pseudo-likelihood conditioning.")
    if conditioning == "whole_entity" and entities is None:
        raise ValueError("Whole-entity PLL requires explicit predicted entity occurrences.")
    device = next(model.parameters()).device
    nll = 0.0
    targets = occurrences = skipped = 0
    with bidirectional_attention(model=model), torch.inference_mode():
        model.eval()
        for window in windows:
            groups = [word.positions for word in window.words]
            if conditioning == "whole_entity":
                groups = []
                for span in (entities or {}).get(window.doc_id, ()):
                    words = [
                        word
                        for word in window.words
                        if span.start <= word.start and word.end <= span.end
                    ]
                    if not words or words[0].start != span.start or words[-1].end != span.end:
                        if window.offsets[-2][1] > span.start and window.offsets[1][0] < span.end:
                            skipped += 1
                        continue
                    groups.append(tuple(position for word in words for position in word.positions))
            for group in groups:
                occurrences += 1
                for index, position in enumerate(group):
                    masked = group[index:] if conditioning == "within_word_l2r" else group
                    ids = torch.tensor([window.input_ids], device=device)
                    ids[0, list(masked)] = tokenizer.mask_token_id
                    logits = (
                        model(input_ids=ids, attention_mask=torch.ones_like(ids))
                        .logits[0, position]
                        .float()
                    )
                    nll -= float(logits.log_softmax(-1)[window.input_ids[position]])
                    targets += 1
    if not targets:
        raise ValueError("Pseudo-likelihood has zero valid targets; check aligned spans.")

    return {
        "mean_pll": -nll / targets,
        "pppl": math.exp(nll / targets),
        "targets": targets,
        "occurrences": occurrences,
        "skipped_spans": skipped,
    }
