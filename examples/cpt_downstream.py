"""Compose frozen CPT probes with canonical NEL/NER metrics outside task subpackages."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Literal

import pandas as pd
import torch

from lab.core.scoring import score_spans
from lab.nel.evaluation.metrics import retrieval_metrics_from_codes


def frozen_linking_probe(
    *,
    model: Any,
    tokenizer: Any,
    terminology: pd.DataFrame,
    mentions: pd.DataFrame,
    mode: Literal["mention_only", "contextual"] = "mention_only",
    seen_concepts: frozenset[str] = frozenset(),
    seen_surfaces: frozenset[str] = frozenset(),
    k_values: Sequence[int] = (1, 5, 25),
) -> dict[str, float]:
    """Rebuild normalized alias vectors for this checkpoint and deduplicate by concept.

    Contextual mentions pool only subwords overlapping the supplied exact mention span.
    Inputs are caller-selected dev tables. Final-test data are a separate explicit call.
    This is a frozen representation probe, with no downstream fine-tuning.
    """
    if mode not in {"mention_only", "contextual"}:
        raise ValueError("mode must be mention_only or contextual.")
    if terminology.empty or mentions.empty:
        raise ValueError("Supply a nonempty fixed terminology and mention table.")
    device = next(model.parameters()).device
    was_training = model.training
    model.eval()
    encoder = model.base_model

    def embed(text: str, start: int | None = None, end: int | None = None) -> torch.Tensor:
        encoded = tokenizer(
            text, return_offsets_mapping=True, return_tensors="pt", truncation=False
        )
        offsets = encoded.pop("offset_mapping")[0].tolist()
        selected = [
            index
            for index, (left, right) in enumerate(offsets)
            if right > left and (start is None or (left < end and right > start))
        ]
        if not selected:
            raise ValueError("Mention has no aligned subwords; repair offsets.")
        hidden = encoder(
            **{key: value.to(device) for key, value in encoded.items()}
        ).last_hidden_state
        return torch.nn.functional.normalize(hidden[0, selected].mean(0).float(), dim=-1)

    try:
        with torch.inference_mode():
            candidates = torch.stack([embed(row.term) for row in terminology.itertuples()])
            codes = terminology.code.astype(str).tolist()
            predictions = []
            margins = []
            for row in mentions.itertuples():
                vector = (
                    embed(row.text)
                    if mode == "mention_only"
                    else embed(row.context, row.start_span, row.end_span)
                )
                scores = candidates @ vector
                best: dict[str, float] = {}
                for code, score in zip(codes, scores.tolist(), strict=True):
                    best[code] = max(best.get(code, -float("inf")), score)
                ranked = sorted(best, key=lambda code: (-best[code], code))
                predictions.append(ranked)
                gold_score = best.get(str(row.code))
                if gold_score is not None and any(code != str(row.code) for code in ranked):
                    margins.append(
                        gold_score - max(best[code] for code in ranked if code != str(row.code))
                    )
    finally:
        model.train(was_training)
    gold = mentions.code.astype(str).tolist()
    result = dict(
        retrieval_metrics_from_codes(
            gold_codes=gold, predicted_codes=predictions, k_values=k_values
        )
    )
    if margins:
        result["ranking_margin"] = sum(margins) / len(margins)
    for name, subset in (
        ("seen_concept", [code in seen_concepts for code in gold]),
        ("unseen_concept", [code not in seen_concepts for code in gold]),
        ("seen_surface", [text.casefold() in seen_surfaces for text in mentions.text]),
        ("unseen_surface", [text.casefold() not in seen_surfaces for text in mentions.text]),
    ):
        indices = [index for index, selected in enumerate(subset) if selected]
        if indices:
            metrics = retrieval_metrics_from_codes(
                gold_codes=[gold[index] for index in indices],
                predicted_codes=[predictions[index] for index in indices],
                k_values=k_values,
            )
            result.update({f"{name}/{key}": value for key, value in metrics.items()})

    return result


def exact_ner_metrics(*, predicted: pd.DataFrame, gold: pd.DataFrame) -> dict[str, Any]:
    """Reuse the canonical exact labelled entity scorer; never substitute token accuracy."""
    return score_spans(
        gold=gold, predicted=predicted, tags=sorted(set(gold.label) | set(predicted.label))
    )
