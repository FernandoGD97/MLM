"""Precompute aligned importance outside dynamic batch corruption."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import pandas as pd

from lab.cpt.contracts import ImportanceResult, Span, Window
from lab.cpt.importance.domain import DomainScores
from lab.cpt.importance.entities import score_entities
from lab.cpt.importance.keyphrases import extract_keyphrases
from lab.cpt.importance.projection import project_phrase_scores
from lab.cpt.masking.policies import MaskPolicy, WindowSignals


def prepare_signals(
    *,
    documents: pd.DataFrame,
    windows: Sequence[Window],
    policy: MaskPolicy,
    entity_spans: pd.DataFrame | None = None,
    domain_scores: DomainScores | None = None,
    entity_mode: str = "binary",
    type_weights: Mapping[str, float] | None = None,
) -> dict[tuple[str, int], WindowSignals]:
    """Build word-occurrence signals from independent predicted spans and fitted statistics."""
    needs_entities = policy.policy in {"M1", "M2", "M5", "M6", "M7"}
    needs_domain = policy.policy in {"M3", "M4", "M5", "M6", "M7"}
    if needs_entities and entity_spans is None:
        raise ValueError(f"{policy.policy} requires precomputed predicted entity spans.")
    if needs_domain and domain_scores is None:
        raise ValueError(f"{policy.policy} requires training-fitted domain word statistics.")
    lookup = dict(domain_scores.scores) if domain_scores else {}
    entities: dict[str, ImportanceResult] = {}
    keyphrases: dict[str, ImportanceResult] = {}
    for row in documents.itertuples():
        if needs_entities:
            frame = entity_spans[entity_spans.filename == row.doc_id]
            entities[row.doc_id] = score_entities(
                spans=frame, text=row.text, mode=entity_mode, type_weights=type_weights
            )
        if needs_domain:
            keyphrases[row.doc_id] = extract_keyphrases(text=row.text, word_scores=lookup)
    result = {}
    for window in windows:
        entity_result, skipped = _complete_spans(window, entities.get(window.doc_id))
        keyphrase_result, _ = _complete_spans(window, keyphrases.get(window.doc_id))
        entity = project_phrase_scores(
            words=window.words, importance=entity_result, method="maximum"
        ).scores
        phrase = project_phrase_scores(
            words=window.words, importance=keyphrase_result, method=policy.phrase_projection
        ).scores
        result[(window.doc_id, window.index)] = WindowSignals(
            entity=entity,
            domain=tuple(lookup.get(word.text.casefold(), 0.0) for word in window.words),
            keyphrase=phrase,
            entities=entity_result.spans,
            skipped_spans=skipped,
        )

    return result


def _complete_spans(
    window: Window, importance: ImportanceResult | None
) -> tuple[ImportanceResult, int]:
    spans: list[Span] = []
    skipped = 0
    if importance:
        starts = {word.start for word in window.words}
        ends = {word.end for word in window.words}
        left = min(starts, default=0)
        right = max(ends, default=0)
        for span in importance.spans:
            if span.end <= left or span.start >= right:
                continue
            if span.start not in starts or span.end not in ends:
                skipped += 1
            else:
                spans.append(span)

    return ImportanceResult(
        tuple(span.score for span in spans),
        tuple(spans),
        "span_occurrence",
        "complete_window_spans",
    ), skipped
