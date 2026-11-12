"""Immutable occurrence-level contracts; character offsets are half-open."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True)
class Span:
    """A scored character occurrence with optional entity provenance."""

    start: int
    end: int
    score: float = 1.0
    label: str | None = None
    concept: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.start, int) or not isinstance(self.end, int):
            raise ValueError("Span offsets must be integer character positions.")
        if self.start < 0 or self.end <= self.start or not math.isfinite(self.score):
            raise ValueError("Span requires 0 <= start < end and a finite score.")


@dataclass(frozen=True)
class Word:
    """A UAX #29 word occurrence aligned to positions inside one model window."""

    start: int
    end: int
    text: str
    positions: tuple[int, ...]


@dataclass(frozen=True)
class Window:
    """One document-bounded window; offsets retain document coordinates."""

    doc_id: str
    index: int
    input_ids: tuple[int, ...]
    offsets: tuple[tuple[int, int], ...]
    words: tuple[Word, ...]
    eligible: tuple[int, ...]


@dataclass(frozen=True)
class ImportanceResult:
    """Scores with an explicit unit, occurrence spans and reproducible parameters."""

    scores: tuple[float, ...]
    spans: tuple[Span, ...]
    unit: Literal["word_occurrence", "word_type", "span_occurrence", "token_position"]
    method: str
    parameters: tuple[tuple[str, str], ...] = ()

    def __post_init__(self) -> None:
        if not all(math.isfinite(score) for score in self.scores):
            raise ValueError("Importance scores must be finite; resolve missing values explicitly.")
        if self.unit in {"word_occurrence", "span_occurrence"}:
            if len(self.scores) != len(self.spans):
                raise ValueError("Occurrence scores and spans must have the same length.")
