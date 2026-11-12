"""Explicit statistics and masked-text artifacts, retaining original corpus distribution."""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import asdict
from pathlib import Path

import pandas as pd

from lab.core.sentences import describe_sentence_segmenter
from lab.cpt.inspection.corpus import Document
from lab.cpt.inspection.selection import Selection


def corpus_statistics(*, documents: Sequence[Document], paths: Sequence[str]) -> dict:
    """Count original documents, sentences, eligible words and annotated occurrences."""
    rows = []
    for corpus, path in enumerate(paths):
        subset = [doc for doc in documents if doc.corpus == corpus]
        rows.append(
            {
                "corpus": corpus,
                "path": str(path),
                "documents": len(subset),
                "sentences": sum(len(doc.sentences) for doc in subset),
                "words": sum(len(doc.words) for doc in subset),
                "characters": sum(len(doc.text) for doc in subset),
                "entities": sum(len(doc.entities) for doc in subset),
                "languages": sorted({doc.language for doc in subset}),
            }
        )
    total = {
        key: sum(row[key] for row in rows)
        for key in ("documents", "sentences", "words", "characters", "entities")
    }
    segmenters = {
        code: describe_sentence_segmenter(code)
        for code in sorted({doc.language for doc in documents})
    }

    return {"total": total, "corpora": rows, "segmenters": segmenters}


def masked_frames(
    *,
    frames: Sequence[pd.DataFrame],
    documents: Sequence[Document],
    selection: Selection,
    mask_token: str = "[MASK]",
) -> tuple[pd.DataFrame, ...]:
    """Add masked text, selected word offsets and counts without rewriting original fields."""
    selected = set(selection.positions)
    results = [frame.copy() for frame in frames]
    payloads: list[list[dict]] = [[] for _ in frames]
    offset = 0
    for doc in documents:
        spans = [span for index, span in enumerate(doc.words) if offset + index in selected]
        offset += len(doc.words)
        masked = doc.text
        for start, end in reversed(spans):
            masked = masked[:start] + mask_token + masked[end:]
        payloads[doc.corpus].append(
            {
                "masked_text": masked,
                "n_masked_words": len(spans),
                "masked_word_spans_json": json.dumps(spans),
            }
        )
    for result, rows in zip(results, payloads, strict=True):
        for name in ("masked_text", "n_masked_words", "masked_word_spans_json"):
            result[name] = [row[name] for row in rows]

    return tuple(results)


def write_masking(
    *,
    output_dir: Path,
    technique: str,
    frames: Sequence[pd.DataFrame],
    selection: Selection,
    total_words: int,
) -> dict:
    """Write separate per-source outputs and the realized global/proportional word budget."""
    target = output_dir / technique
    target.mkdir()
    report = asdict(selection)
    report.pop("positions")
    report.update(
        selected_words=len(selection.positions),
        eligible_words=total_words,
        effective_probability=len(selection.positions) / total_words if total_words else 0,
        deviation=len(selection.positions) - selection.requested,
    )
    for index, frame in enumerate(frames):
        frame.to_parquet(target / f"corpus_{index:03d}.parquet", index=False)
    (target / "summary.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    return report
