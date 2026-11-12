"""Percentile normalization and explicit weighted score/rank fusion."""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Literal


def percentile_scores(*, scores: Sequence[float]) -> tuple[float, ...]:
    """Return midrank percentiles, with all-zero evidence kept at zero."""
    if not all(math.isfinite(value) for value in scores):
        raise ValueError("Resolve nonfinite importance before normalization.")
    if not scores or not any(scores):
        return tuple(0.0 for _ in scores)

    return tuple(
        0.0
        if value == 0
        else (
            sum(other < value for other in scores) + 0.5 * sum(other == value for other in scores)
        )
        / len(scores)
        for value in scores
    )


def fuse_scores(
    *,
    signals: Sequence[Sequence[float]],
    weights: Sequence[float],
    method: Literal["score", "weighted_rank", "rrf"] = "score",
    rrf_constant: float = 60,
) -> tuple[float, ...]:
    """Fuse aligned word occurrences; caller-specified weights are experimental choices."""
    if method not in {"score", "weighted_rank", "rrf"}:
        raise ValueError(f"Unknown fusion method {method!r}.")
    if not signals or len(signals) != len(weights) or len({len(row) for row in signals}) != 1:
        raise ValueError("Provide aligned signals and one weight per signal.")
    if not all(math.isfinite(weight) and weight >= 0 for weight in weights) or sum(weights) <= 0:
        raise ValueError("Fusion weights must be finite, nonnegative, and not all zero.")
    if not math.isfinite(rrf_constant) or rrf_constant <= 0:
        raise ValueError("rrf_constant must be positive and finite.")
    fused = [0.0] * len(signals[0])
    for signal, weight in zip(signals, weights, strict=True):
        normalized = percentile_scores(scores=signal)
        ranks = [1 + sum(other > value for other in signal) for value in signal]
        for index, value in enumerate(normalized):
            if signal[index] == 0:
                continue
            contribution = (
                value
                if method == "score"
                else 1 / (ranks[index] + (rrf_constant if method == "rrf" else 0))
            )
            fused[index] += weight * contribution / sum(weights)

    return tuple(fused)
