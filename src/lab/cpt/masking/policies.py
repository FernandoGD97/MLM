"""Discoverable masking and replacement parameters shared by Python and YAML."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

from lab.cpt.contracts import Span


@dataclass(frozen=True)
class MaskPolicy:
    """Scientific choices for target selection, independent of token corruption."""

    policy: Literal["M0", "M1", "M2", "M3", "M4", "M5", "M6", "M7", "M8"] = "M0"
    probability: float = 0.15
    random_unit: Literal["subword", "word"] = "subword"
    budget_mode: Literal["strict", "underfill"] = "strict"
    fusion: Literal["score", "weighted_rank", "rrf", "hierarchical", "disjoint"] = "score"
    weights: tuple[float, float, float] = (1.0, 1.0, 1.0)
    phrase_projection: Literal["maximum", "mean", "sum", "uniform", "length_normalized"] = "maximum"
    mask_seed: int = 43
    mask_fraction: float = 0.8
    random_fraction: float = 0.1
    adaptive_enabled: bool = False
    adaptive_refresh_epochs: int = 1

    def __post_init__(self) -> None:
        object.__setattr__(self, "weights", tuple(self.weights))
        if self.policy not in {f"M{i}" for i in range(9)}:
            raise ValueError(f"Unknown mask policy {self.policy!r}.")
        if not math.isfinite(self.probability) or not 0 <= self.probability <= 1:
            raise ValueError("Mask probability must be within [0, 1].")
        if self.random_unit not in {"subword", "word"} or self.budget_mode not in {
            "strict",
            "underfill",
        }:
            raise ValueError("Use random_unit=subword/word and budget_mode=strict/underfill.")
        if self.fusion not in {"score", "weighted_rank", "rrf", "hierarchical", "disjoint"}:
            raise ValueError("Unknown fusion method.")
        if (
            len(self.weights) != 3
            or not all(math.isfinite(value) and value >= 0 for value in self.weights)
            or sum(self.weights) <= 0
        ):
            raise ValueError(
                "Provide three finite nonnegative fusion weights, at least one positive."
            )
        if (
            not all(
                math.isfinite(value) and value >= 0
                for value in (self.mask_fraction, self.random_fraction)
            )
            or self.mask_fraction + self.random_fraction > 1
        ):
            raise ValueError("Replacement fractions must be nonnegative and sum to at most one.")
        if self.policy == "M8" and not self.adaptive_enabled:
            raise ValueError("M8 requires adaptive_enabled=True explicitly.")
        if self.adaptive_refresh_epochs < 1:
            raise ValueError("adaptive_refresh_epochs must be >= 1.")


@dataclass(frozen=True)
class WindowSignals:
    """Aligned word-occurrence scores and complete entity character occurrences."""

    entity: tuple[float, ...] = ()
    domain: tuple[float, ...] = ()
    keyphrase: tuple[float, ...] = ()
    uncertainty: tuple[float, ...] = ()
    entities: tuple[Span, ...] = ()
    skipped_spans: int = 0
