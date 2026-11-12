"""Local exact-occurrence keyphrases ranked by training-fitted lexical relevance."""

from __future__ import annotations

from collections.abc import Mapping

from lab.core.segmentation import split_into_words
from lab.cpt.contracts import ImportanceResult, Span


def extract_keyphrases(
    *, text: str, word_scores: Mapping[str, float], maximum_words: int = 3, top_k: int = 10
) -> ImportanceResult:
    """Rank contiguous alphanumeric n-grams by mean fitted word score.

    Candidates retain exact offsets, including repeated phrases. This inexpensive lexical
    extractor makes no claim to match embedding-based or generative extraction methods.
    """
    if maximum_words < 1 or top_k < 1:
        raise ValueError("maximum_words and top_k must be positive.")
    words = split_into_words(text)
    candidates = []
    for start in range(len(words)):
        for length in range(1, min(maximum_words, len(words) - start) + 1):
            group = words[start : start + length]
            if not all(any(character.isalnum() for character in word["text"]) for word in group):
                continue
            score = sum(word_scores.get(word["text"].casefold(), 0.0) for word in group) / length
            if score > 0:
                candidates.append(Span(group[0]["start"], group[-1]["end"], score))
    chosen = sorted(candidates, key=lambda span: (-span.score, span.start, span.end))[:top_k]

    return ImportanceResult(
        tuple(span.score for span in chosen),
        tuple(chosen),
        "span_occurrence",
        "lexical_ngram_mean",
        (("maximum_words", str(maximum_words)), ("top_k", str(top_k))),
    )
