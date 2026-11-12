"""Predicted entity spans from canonical span tables, without loading a detector."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Literal

import pandas as pd

from lab.cpt.contracts import ImportanceResult, Span


def score_entities(
    *,
    spans: pd.DataFrame,
    text: str,
    mode: Literal["binary", "confidence"] = "binary",
    type_weights: Mapping[str, float] | None = None,
    concept_frequencies: Mapping[str, int] | None = None,
) -> ImportanceResult:
    """Score predicted occurrences; confidence and type/frequency weighting are explicit.

    Frequency weighting is inverse square-root frequency and requires reliable concept IDs.
    The caller selects one document from the canonical filename/start_span/end_span table.
    """
    if mode not in {"binary", "confidence"}:
        raise ValueError("Entity mode must be binary or confidence.")
    result = []
    for row in spans.to_dict("records"):
        start, end = row["start_span"], row["end_span"]
        if (
            not isinstance(start, int)
            or not isinstance(end, int)
            or not 0 <= start < end <= len(text)
        ):
            raise ValueError("Predicted entity offsets must be valid half-open character spans.")
        if row.get("text", text[start:end]) != text[start:end]:
            raise ValueError("Predicted entity text does not match offsets; repair the span table.")
        score = 1.0 if mode == "binary" else float(row["score"])
        if mode == "confidence" and not 0 <= score <= 1:
            raise ValueError("Entity confidence must be within [0, 1].")
        score *= (type_weights or {}).get(row["label"], 1.0)
        concept = row.get("code")
        concept = concept if isinstance(concept, str) and concept else None
        if concept_frequencies is not None:
            if concept is None or concept_frequencies.get(concept, 0) < 1:
                raise ValueError("Frequency weighting requires a known concept and positive count.")
            score /= concept_frequencies[concept] ** 0.5
        result.append(Span(start, end, score, row["label"], concept))

    return ImportanceResult(
        tuple(span.score for span in result),
        tuple(result),
        "span_occurrence",
        f"ner_{mode}",
        (
            ("type_weights", str(sorted((type_weights or {}).items()))),
            ("concept_frequencies", str(sorted((concept_frequencies or {}).items()))),
        ),
    )


def read_entity_spans(*, path: str | Path) -> pd.DataFrame:
    """Read precomputed predictions using the canonical span columns."""
    frame = (
        pd.read_parquet(path)
        if str(path).endswith(".parquet")
        else pd.read_csv(path, sep="\t", keep_default_na=False)
    )
    required = {"filename", "label", "start_span", "end_span", "text"}
    if not required <= set(frame):
        raise ValueError(f"Entity predictions lack columns: {sorted(required - set(frame))}.")

    return frame
