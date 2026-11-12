"""Canonical unannotated corpus preparation and document-bounded alignment."""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd

from lab.core.segmentation import split_into_words
from lab.cpt.contracts import Window, Word

if TYPE_CHECKING:
    from transformers import PreTrainedTokenizerBase


def prepare_corpus(
    *, documents: pd.DataFrame | str | Path, output_dir: str | Path | None = None
) -> pd.DataFrame:
    """Validate raw text, discard gold annotations and retain supplied metadata.

    Accept a Parquet path, DataFrame or Hugging Face Dataset via its `to_pandas` method.
    Missing source/language/translation metadata remain missing. Disk output is explicit.
    """
    if isinstance(documents, (str, Path)):
        frame = pd.read_parquet(documents)
    elif isinstance(documents, pd.DataFrame):
        frame = documents.copy()
    elif hasattr(documents, "to_pandas"):
        frame = documents.to_pandas()
    else:
        raise TypeError("documents must be a DataFrame, Dataset or Parquet path.")
    if not {"doc_id", "text"} <= set(frame):
        raise ValueError("Corpus requires doc_id and text columns.")
    for column in ("doc_id", "text"):
        if not all(isinstance(value, str) and value.strip() for value in frame[column]):
            raise ValueError(f"{column} requires nonempty strings; cast identifiers explicitly.")
    if frame.doc_id.duplicated().any():
        raise ValueError("doc_id must be unique; disambiguate sources before preparation.")
    for column in ("source", "language", "translation_status", "original_doc_id"):
        if column in frame and not all(
            pd.isna(value) or isinstance(value, str) for value in frame[column]
        ):
            raise ValueError(f"{column} must contain strings or missing values.")
    frame["entities_json"] = "[]"
    frame["n_entities"] = pd.Series(0, index=frame.index, dtype="int32")
    if output_dir is not None:
        target = Path(output_dir)
        target.mkdir(parents=True, exist_ok=False)
        frame.to_parquet(target / "corpus.parquet", index=False)

    return frame.reset_index(drop=True)


def grouped_split(
    *, documents: pd.DataFrame, validation_fraction: float = 0.2, random_state: int = 42
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split connected original/translation families together, including chained IDs."""
    import random

    if not 0 < validation_fraction < 1:
        raise ValueError("validation_fraction must be between zero and one.")
    frame = prepare_corpus(documents=documents)
    parent: dict[str, str] = {}

    def find(key: str) -> str:
        parent.setdefault(key, key)
        if parent[key] != key:
            parent[key] = find(parent[key])
        return parent[key]

    for row in frame.to_dict("records"):
        original = row.get("original_doc_id")
        if isinstance(original, str) and original:
            parent[find(row["doc_id"])] = find(original)
    groups = sorted({find(key) for key in frame.doc_id})
    if len(groups) < 2:
        raise ValueError("At least two unrelated document families are needed for a split.")
    random.Random(random_state).shuffle(groups)
    count = max(1, min(len(groups) - 1, int(len(groups) * validation_fraction)))
    validation = set(groups[:count])
    selected = frame.doc_id.map(lambda key: find(key) in validation)

    return frame[~selected].reset_index(drop=True), frame[selected].reset_index(drop=True)


def tokenize_corpus(
    *, documents: pd.DataFrame, tokenizer: PreTrainedTokenizerBase, max_length: int = 128
) -> tuple[Window, ...]:
    """Tokenize once per document, retaining full words and original character offsets.

    Special tokens use zero offsets. Words larger than the window and tokens straddling
    UAX word boundaries fail explicitly instead of silently fragmenting whole-word units.
    """
    if not tokenizer.is_fast:
        raise ValueError("Word/span policies require a fast tokenizer with reliable offsets.")
    capacity = max_length - tokenizer.num_special_tokens_to_add(pair=False)
    if capacity < 1:
        raise ValueError("max_length must leave room for at least one content token.")
    windows = []
    for row in prepare_corpus(documents=documents).itertuples():
        encoded = tokenizer(row.text, add_special_tokens=False, return_offsets_mapping=True)
        ids, offsets = encoded["input_ids"], encoded["offset_mapping"]
        words = split_into_words(row.text)
        assigned: list[int | None] = []
        for start, end in offsets:
            if not 0 <= start < end <= len(row.text):
                raise ValueError("Tokenizer returned invalid offsets; use a compatible tokenizer.")
            while start < end and row.text[start].isspace():
                start += 1
            matches = [
                i for i, word in enumerate(words) if start < word["end"] and end > word["start"]
            ]
            if len(matches) > 1:
                raise ValueError("A subword straddles UAX word boundaries; use another tokenizer.")
            assigned.append(matches[0] if matches else None)
        start = 0
        while start < len(ids):
            end = min(start + capacity, len(ids))
            while end < len(ids) and end > start and assigned[end] == assigned[end - 1]:
                end -= 1
            if end == start:
                raise ValueError("One word exceeds max_length; increase the window size.")
            chunk = ids[start:end]
            character_start = offsets[start][0]
            character_end = offsets[end - 1][1]
            prepared = tokenizer(
                row.text[character_start:character_end],
                add_special_tokens=True,
                return_offsets_mapping=True,
                return_special_tokens_mask=True,
            )
            special = prepared["special_tokens_mask"]
            positions = [i for i, flag in enumerate(special) if not flag]
            if [prepared["input_ids"][i] for i in positions] != chunk:
                raise ValueError(
                    "Window retokenization changes document tokens; increase max_length "
                    "or use a boundary-stable tokenizer."
                )
            window_offsets = [
                (0, 0) if flag else (left + character_start, right + character_start)
                for flag, (left, right) in zip(special, prepared["offset_mapping"], strict=True)
            ]
            aligned = []
            for index in sorted({value for value in assigned[start:end] if value is not None}):
                word = words[index]
                word_positions = tuple(
                    positions[j]
                    for j, word_id in enumerate(assigned[start:end])
                    if word_id == index
                )
                aligned.append(Word(**word, positions=word_positions))
            eligible = tuple(position for word in aligned for position in word.positions)
            windows.append(
                Window(
                    row.doc_id,
                    start,
                    tuple(prepared["input_ids"]),
                    tuple(window_offsets),
                    tuple(aligned),
                    eligible,
                )
            )
            start = end

    return tuple(windows)


def corpus_fingerprint(*, documents: pd.DataFrame) -> str:
    """Hash content and metadata without recording text in the manifest."""
    import hashlib

    content = documents.to_json(orient="records", force_ascii=False, default_handler=str)

    return hashlib.sha256(json.dumps(json.loads(content), sort_keys=True).encode()).hexdigest()
