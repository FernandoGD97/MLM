"""CLI-addressable corpus statistics and opt-in comparative word-masking exploration."""

from __future__ import annotations

import json
import warnings
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from lab.cpt.inspection.corpus import read_corpora, segment_corpora
from lab.cpt.inspection.reporting import corpus_statistics, masked_frames, write_masking
from lab.cpt.inspection.selection import ALIASES, TECHNIQUES, select_words
from lab.cpt.inspection.signals import entropy_signals, word_signals


@dataclass(frozen=True)
class InspectionResult:
    """Locations of aggregate statistics and optional per-technique masking summaries."""

    output_dir: Path
    statistics: Path
    techniques: tuple[str, ...]


def inspect_corpora(
    *,
    corpora: Sequence[str],
    output_dir: str,
    language: str | None = None,
    languages: Sequence[str] | None = None,
    techniques: Sequence[str] = (),
    terms: Sequence[str] = (),
    probability: float = 0.15,
    random_state: int = 42,
    case_sensitive: bool = False,
    backfill: bool = True,
    budget_mode: str = "strict",
    show_examples: bool = False,
    examples: int = 3,
    example_characters: int = 300,
    base_model: str | None = None,
) -> InspectionResult:
    """Read, segment, report, then optionally compare masks under a corpus-wide budget.

    This exploratory workflow preserves canonical annotations and never trains. Annotation
    masking is explicitly requested with entities/whole_entities or their fusion variants.
    Example printing is opt-in because partially masked text can still contain clinical PHI.
    """
    import math

    chosen = tuple(ALIASES.get(name, name) for name in techniques)
    if isinstance(corpora, str) or isinstance(techniques, str):
        raise ValueError("corpora and techniques must be lists, not strings.")
    if len(set(chosen)) != len(chosen) or any(name not in TECHNIQUES for name in chosen):
        raise ValueError(f"Choose distinct techniques from {TECHNIQUES} or M0–M8.")
    if not math.isfinite(probability) or not 0 <= probability <= 1:
        raise ValueError("probability must be within [0, 1], e.g. 0.2 for 20%.")
    if examples < 0 or example_characters < 1:
        raise ValueError("examples must be nonnegative and example_characters positive.")
    if "entropy" in chosen and base_model is None:
        raise ValueError("M8/entropy needs an explicit local base_model; it is never simulated.")
    if "terms" in chosen and not terms:
        raise ValueError("Term masking requires terms or a terms file in the driver script.")
    output = Path(output_dir)
    if output.exists():
        raise ValueError("output_dir already exists; choose a fresh directory.")
    frames = read_corpora(paths=corpora)
    documents = segment_corpora(frames=frames, language=language, languages=languages)
    if not documents:
        raise ValueError("Input corpora contain no documents.")
    if any(
        name in {"entities", "whole_entities", "ner_domain", "ner_keyphrase", "triple"}
        for name in chosen
    ) and not any(doc.entities for doc in documents):
        warnings.warn(
            "No annotations supplied: entity signals are zero; only other signals "
            "and explicitly enabled backfill can select words.",
            UserWarning,
            stacklevel=2,
        )
    statistics = corpus_statistics(documents=documents, paths=corpora)
    output.mkdir(parents=True)
    statistics_path = output / "statistics.json"
    statistics_path.write_text(json.dumps(statistics, indent=2), encoding="utf-8")
    pd.DataFrame(
        [
            {
                "corpus": doc.corpus,
                "row": doc.row,
                "doc_id": doc.doc_id,
                "language": doc.language,
                "sentence_spans_json": json.dumps(doc.sentences),
                "word_spans_json": json.dumps(doc.words),
            }
            for doc in documents
        ]
    ).to_parquet(output / "segmentation.parquet", index=False)
    print(json.dumps(statistics, indent=2, ensure_ascii=False))
    if not chosen:
        return InspectionResult(output, statistics_path, chosen)
    evidence = word_signals(documents=documents, terms=terms, case_sensitive=case_sensitive)
    entropy = (
        entropy_signals(documents=documents, base_model=base_model) if "entropy" in chosen else None
    )
    if show_examples:
        warnings.warn(
            "Masking examples may contain clinical PHI; display was explicitly enabled.",
            UserWarning,
            stacklevel=2,
        )
    for technique in chosen:
        selection = select_words(
            documents=documents,
            signals=evidence,
            technique=technique,
            probability=probability,
            random_state=random_state,
            backfill=backfill,
            budget_mode=budget_mode,
            entropy=entropy,
        )
        masked = masked_frames(frames=frames, documents=documents, selection=selection)
        report = write_masking(
            output_dir=output,
            technique=technique,
            frames=masked,
            selection=selection,
            total_words=statistics["total"]["words"],
        )
        print(technique, json.dumps(report))
        if show_examples:
            count = 0
            for corpus, frame in enumerate(masked):
                for row in frame[frame.n_masked_words > 0].itertuples():
                    if count >= examples:
                        break
                    print(
                        f"corpus={corpus} masked_words={row.n_masked_words}: "
                        f"{row.masked_text[:example_characters]}"
                    )
                    count += 1
    configuration = {
        "techniques": chosen,
        "probability": probability,
        "random_state": random_state,
        "terms": list(terms),
        "case_sensitive": case_sensitive,
        "backfill": backfill,
        "budget_mode": budget_mode,
        "language": language,
        "languages": languages,
        "base_model": base_model,
    }
    (output / "masking_config.json").write_text(json.dumps(configuration, indent=2))

    return InspectionResult(output, statistics_path, chosen)
