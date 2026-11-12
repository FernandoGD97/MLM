"""Read canonical Parquets without discarding annotations, metadata or source identity."""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from lab.core.corpus import validate_corpus
from lab.core.segmentation import split_into_words
from lab.core.sentences import sentence_segmenter, split_into_sentences


@dataclass(frozen=True)
class Document:
    """Document identity and original half-open word/sentence character extents."""

    corpus: int
    row: int
    doc_id: str
    text: str
    language: str
    words: tuple[tuple[int, int], ...]
    sentences: tuple[tuple[int, int], ...]
    entities: tuple[tuple[int, int], ...]


def read_corpora(*, paths: Sequence[str | Path]) -> tuple[pd.DataFrame, ...]:
    """Validate each canonical corpus separately, preserving repeated IDs across sources."""
    if not paths:
        raise ValueError("Supply at least one Parquet path.")
    frames = []
    for path in paths:
        frame = pd.read_parquet(path)
        if frame.empty:
            raise ValueError(f"{path}: empty corpus; remove it from the input list.")
        if "entities_json" not in frame:
            if "n_entities" in frame and frame.n_entities.fillna(0).ne(0).any():
                raise ValueError(f"{path}: n_entities is present without entity offsets.")
            frame["entities_json"] = "[]"
        if "n_entities" not in frame:
            frame["n_entities"] = frame.entities_json.map(lambda value: len(json.loads(value)))
        validate_corpus(frame)
        for row in frame.itertuples():
            if not isinstance(row.doc_id, str) or not isinstance(row.text, str):
                raise ValueError(f"{path}: doc_id and text must be strings.")
            if len(json.loads(row.entities_json)) != row.n_entities:
                raise ValueError(f"{path}: n_entities does not match entities_json.")
        frames.append(frame)

    return tuple(frames)


def segment_corpora(
    *,
    frames: Sequence[pd.DataFrame],
    language: str | None = None,
    languages: Sequence[str] | None = None,
) -> tuple[Document, ...]:
    """Segment using row language, per-corpus language, or an explicit fallback.

    All ISO 639-1 languages accepted by core are supported; language is never inferred
    from a filename. Eligible words contain a Unicode letter or number; punctuation-only
    UAX segments remain unmasked. Offsets stay in the untouched original document.
    """
    if languages is not None and len(languages) != len(frames):
        raise ValueError("languages must contain one ISO 639-1 code per input corpus.")
    documents = []
    for corpus, frame in enumerate(frames):
        for index, row in enumerate(frame.to_dict("records")):
            code = row.get("language")
            if not isinstance(code, str) or not code:
                code = languages[corpus] if languages is not None else language
            if code is None:
                raise ValueError("Missing language: supply row language, languages, or language.")
            sentence_segmenter(code)
            text = row["text"]
            words = tuple(
                (word["start"], word["end"])
                for word in split_into_words(text)
                if any(character.isalnum() for character in word["text"])
            )
            sentences = tuple(
                (span["start"], span["end"]) for span in split_into_sentences(text, code)
            )
            entities = tuple(
                (span["start"], span["end"]) for span in json.loads(row["entities_json"])
            )
            documents.append(
                Document(corpus, index, row["doc_id"], text, code, words, sentences, entities)
            )

    return tuple(documents)
