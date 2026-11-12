"""Explicit phrase-occurrence to UAX word-occurrence score projection."""

from __future__ import annotations

from typing import Literal

from lab.cpt.contracts import ImportanceResult, Span, Word


def project_phrase_scores(
    *,
    words: tuple[Word, ...],
    importance: ImportanceResult,
    method: Literal["maximum", "mean", "sum", "uniform", "length_normalized"],
) -> ImportanceResult:
    """Project aligned full phrases; missing word scores are zero.

    Maximum/mean/sum aggregate full phrase scores over each word's covering occurrences.
    Uniform adds score/word_count. Length-normalized adds score*word_chars/phrase_word_chars.
    Truncated or mid-word spans raise; callers must explicitly discard and count them first.
    """
    if method not in {"maximum", "mean", "sum", "uniform", "length_normalized"}:
        raise ValueError(f"Unknown phrase projection {method!r}.")
    if importance.unit != "span_occurrence":
        raise ValueError("Phrase projection requires span_occurrence importance.")
    values: list[list[float]] = [[] for _ in words]
    for span, score in zip(importance.spans, importance.scores, strict=True):
        selected = [
            i for i, word in enumerate(words) if word.start < span.end and word.end > span.start
        ]
        if (
            not selected
            or words[selected[0]].start != span.start
            or words[selected[-1]].end != span.end
        ):
            raise ValueError("Phrase offsets do not align to complete words; discard explicitly.")
        total_length = sum(words[index].end - words[index].start for index in selected)
        for index in selected:
            contribution = score
            if method == "uniform":
                contribution /= len(selected)
            elif method == "length_normalized":
                contribution *= (words[index].end - words[index].start) / total_length
            values[index].append(contribution)
    scores = []
    for contributions in values:
        score = max(contributions, default=0.0) if method == "maximum" else sum(contributions)
        if method == "mean" and contributions:
            score /= len(contributions)
        scores.append(score)

    return ImportanceResult(
        tuple(scores),
        tuple(Span(word.start, word.end, score) for word, score in zip(words, scores, strict=True)),
        "word_occurrence",
        f"phrase_{method}",
        importance.parameters,
    )
