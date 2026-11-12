"""Occurrence-aligned exploratory signals without modifying corpus annotations."""

from __future__ import annotations

from collections.abc import Sequence

from lab.core.segmentation import split_into_words
from lab.cpt.contracts import Span, Word
from lab.cpt.importance.domain import score_domain_words
from lab.cpt.importance.keyphrases import extract_keyphrases
from lab.cpt.inspection.corpus import Document


def term_spans(*, text: str, terms: Sequence[str], case_sensitive: bool) -> tuple[Span, ...]:
    """Match literal term word sequences with Unicode casefold and original offsets."""
    words = split_into_words(text)
    normalize = (lambda value: value) if case_sensitive else str.casefold
    tokens = [normalize(word["text"]) for word in words]
    result = set()
    for term in terms:
        if not isinstance(term, str) or not term.strip():
            raise ValueError("Initial terms must be nonempty strings.")
        query = [normalize(word["text"]) for word in split_into_words(term)]
        for index in range(len(tokens) - len(query) + 1):
            if tokens[index : index + len(query)] == query:
                result.add((words[index]["start"], words[index + len(query) - 1]["end"]))

    return tuple(Span(start, end) for start, end in sorted(result))


def word_signals(
    *,
    documents: Sequence[Document],
    terms: Sequence[str],
    case_sensitive: bool,
) -> tuple[tuple[tuple[float, ...], ...], ...]:
    """Return term, annotated-entity, domain and keyphrase signals per document.

    Domain statistics fit only this exploratory input. They must not be reused as
    training statistics when the inspected corpus includes dev/test documents.
    """
    fitted = score_domain_words(training_texts=[doc.text for doc in documents], minimum_frequency=1)
    lookup = dict(fitted.scores)
    results = []
    for document in documents:
        words = tuple(
            Word(start, end, document.text[start:end], (index,))
            for index, (start, end) in enumerate(document.words)
        )
        term = term_spans(text=document.text, terms=terms, case_sensitive=case_sensitive)
        entity = tuple(Span(start, end) for start, end in document.entities)
        phrase = extract_keyphrases(text=document.text, word_scores=lookup)
        signals = []
        for spans in (term, entity, phrase.spans):
            signals.append(
                tuple(
                    max(
                        (span.score for span in spans if start < span.end and end > span.start),
                        default=0.0,
                    )
                    for start, end in document.words
                )
            )
        domain = tuple(lookup.get(word.text.casefold(), 0.0) for word in words)
        results.append((signals[0], signals[1], domain, signals[2]))

    return tuple(results)


def entropy_signals(
    *,
    documents: Sequence[Document],
    base_model: str,
) -> tuple[tuple[float, ...], ...]:
    """Compute genuine masked entropy with an explicit local checkpoint, on demand."""
    import pandas as pd
    from transformers import AutoModelForMaskedLM, AutoTokenizer

    from lab.cpt.corpus import tokenize_corpus
    from lab.cpt.importance.uncertainty import predictive_entropy
    from lab.cpt.modeling import set_attention_mode

    tokenizer = AutoTokenizer.from_pretrained(base_model, local_files_only=True)
    model = AutoModelForMaskedLM.from_pretrained(base_model, local_files_only=True)
    set_attention_mode(model=model, objective="mlm")
    frame = pd.DataFrame(
        [{"doc_id": str(index), "text": doc.text} for index, doc in enumerate(documents)]
    )
    windows = tokenize_corpus(documents=frame, tokenizer=tokenizer)
    cache, _ = predictive_entropy(model=model, tokenizer=tokenizer, windows=windows)
    lookup = {}
    for window in windows:
        for word, score in zip(window.words, cache[(window.doc_id, window.index)], strict=True):
            lookup[(int(window.doc_id), word.start, word.end)] = score

    return tuple(
        tuple(lookup[(index, start, end)] for start, end in document.words)
        for index, document in enumerate(documents)
    )
