"""Explicit corpus-volume matching and aggregate native/translation provenance reports."""

from __future__ import annotations

import random
from collections import Counter
from typing import Any

import pandas as pd

from lab.cpt.corpus import prepare_corpus, tokenize_corpus


def describe_corpus(*, documents: pd.DataFrame) -> dict[str, Any]:
    """Return aggregate metadata/duplication counts without recording text or identifiers."""
    frame = prepare_corpus(documents=documents)
    result: dict[str, Any] = {
        "documents": len(frame),
        "characters": int(frame.text.str.len().sum()),
        "duplicate_text_documents": int(frame.text.duplicated().sum()),
        "original_id_present": int(frame.original_doc_id.notna().sum())
        if "original_doc_id" in frame
        else 0,
    }
    for column in (
        "source",
        "language",
        "translation_status",
        "source_language",
        "target_language",
    ):
        if column in frame:
            result[column] = dict(Counter(str(value) for value in frame[column].dropna()))

    return result


def match_corpus_volume(
    *,
    first: pd.DataFrame,
    second: pd.DataFrame,
    tokenizer: Any,
    token_budget: int,
    max_length: int = 128,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, int]:
    """Match eligible-subword volume exactly using whole-document subset feasibility.

    Select the largest common feasible volume no larger than the explicit token budget.
    This offline dynamic program is intended for bounded research subsets, not streaming
    billion-token corpora. Original/translation family splitting must happen first.
    """
    if token_budget < 1:
        raise ValueError("token_budget must be positive.")
    frames = [prepare_corpus(documents=frame) for frame in (first, second)]
    orders = []
    suffixes = []
    for frame in frames:
        counts: Counter[str] = Counter()
        for window in tokenize_corpus(documents=frame, tokenizer=tokenizer, max_length=max_length):
            counts[window.doc_id] += len(window.eligible)
        order = [(index, counts[row.doc_id]) for index, row in enumerate(frame.itertuples())]
        random.Random(random_state).shuffle(order)
        suffix = [1] * (len(order) + 1)
        limit = (1 << (token_budget + 1)) - 1
        for index in range(len(order) - 1, -1, -1):
            suffix[index] = (suffix[index + 1] | suffix[index + 1] << order[index][1]) & limit
        orders.append(order)
        suffixes.append(suffix)
    common = suffixes[0][0] & suffixes[1][0]
    target = common.bit_length() - 1
    if target < 1:
        raise ValueError("No positive common whole-document volume fits token_budget; increase it.")
    result = []
    for frame, order, suffix in zip(frames, orders, suffixes, strict=True):
        remaining = target
        selected = []
        for index, (row, size) in enumerate(order):
            if size <= remaining and (suffix[index + 1] >> (remaining - size)) & 1:
                selected.append(row)
                remaining -= size
        result.append(frame.iloc[selected].reset_index(drop=True))

    return result[0], result[1], target
