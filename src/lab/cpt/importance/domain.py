"""Training-only lexical statistics for domain discrimination, not clinical importance."""

from __future__ import annotations

import hashlib
import json
import math
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

from lab.core.segmentation import split_into_words


@dataclass(frozen=True)
class DomainScores:
    """Word-type scores and a fingerprint of the fitting documents."""

    scores: tuple[tuple[str, float], ...]
    method: str
    training_sha256: str
    minimum_frequency: int
    smoothing: float


def score_domain_words(
    *,
    training_texts: Sequence[str],
    reference_texts: Sequence[str] = (),
    method: Literal["tfidf", "tficf", "log_odds", "frequency_ratio"] = "tfidf",
    minimum_frequency: int = 2,
    smoothing: float = 0.5,
) -> DomainScores:
    """Fit lexical scores only on explicitly supplied training/reference corpora.

    TF-IDF uses document frequency; TF-ICF uses frequency across the two corpora.
    Log odds uses a symmetric additive prior. Frequency ratio is smoothed relative TF.
    Punctuation-only units are excluded; negations, doses and numbers remain eligible.
    """
    if method not in {"tfidf", "tficf", "log_odds", "frequency_ratio"}:
        raise ValueError(f"Unknown domain method {method!r}.")
    if (
        not training_texts
        or minimum_frequency < 1
        or not math.isfinite(smoothing)
        or smoothing <= 0
    ):
        raise ValueError("Supply training texts, minimum_frequency >= 1 and smoothing > 0.")
    if method != "tfidf" and not reference_texts:
        raise ValueError(f"{method} requires an explicit reference training corpus.")
    documents = [_terms(text) for text in training_texts]
    counts = Counter(word for document in documents for word in document)
    reference = Counter(word for text in reference_texts for word in _terms(text))
    document_frequency = Counter(word for document in documents for word in set(document))
    vocabulary = set(counts) | set(reference)
    size = max(1, len(vocabulary))
    total = sum(counts.values()) + smoothing * (size + 1)
    reference_total = sum(reference.values()) + smoothing * (size + 1)
    scores = []
    for word, count in sorted(counts.items()):
        if count < minimum_frequency:
            continue
        frequency = (count + smoothing) / total
        reference_frequency = (reference[word] + smoothing) / reference_total
        if method == "tfidf":
            score = (
                count
                / max(1, sum(counts.values()))
                * (math.log((1 + len(documents)) / (1 + document_frequency[word])) + 1)
            )
        elif method == "tficf":
            corpus_frequency = 1 + int(reference[word] > 0)
            score = (
                count / max(1, sum(counts.values())) * (math.log(3 / (1 + corpus_frequency)) + 1)
            )
        elif method == "log_odds":
            score = math.log((count + smoothing) / (total - count - smoothing)) - math.log(
                (reference[word] + smoothing) / (reference_total - reference[word] - smoothing)
            )
        else:
            score = frequency / reference_frequency
        scores.append((word, score))
    payload = json.dumps([list(training_texts), list(reference_texts)], ensure_ascii=False)

    return DomainScores(
        tuple(scores),
        method,
        hashlib.sha256(payload.encode()).hexdigest(),
        minimum_frequency,
        smoothing,
    )


def _terms(text: str) -> list[str]:
    return [
        word["text"].casefold()
        for word in split_into_words(text)
        if any(character.isalnum() for character in word["text"])
    ]
