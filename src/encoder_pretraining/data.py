"""Parquet ingestion and document-bounded tokenization (never masking)."""
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Iterable


@dataclass(frozen=True)
class Document:
    doc_id: Any
    text: str
    metadata: dict[str, Any]


@dataclass
class CorpusStats:
    documents: int = 0
    characters: int = 0
    tokens: int = 0
    sequences: int = 0

    @property
    def tokens_per_sequence(self) -> float:
        return self.tokens / self.sequences if self.sequences else 0.0

    def to_dict(self):
        return {**asdict(self), "tokens_per_sequence": self.tokens_per_sequence}


def load_parquet(path: str | list[str], id_column: str, text_columns: list[str]) -> list[Document]:
    """Load non-empty texts and preserve every unused column as opaque metadata."""
    import pyarrow.dataset as ds
    paths = [path] if isinstance(path, str) else path
    table = ds.dataset([str(Path(p)) for p in paths], format="parquet").to_table()
    missing = {id_column, *text_columns} - set(table.column_names)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")
    ignored_annotations = {"entities_json", "n_entities"}
    result = []
    for row in table.to_pylist():
        pieces = [str(row[c]).strip() for c in text_columns if row[c] is not None and str(row[c]).strip()]
        if not pieces:
            continue
        metadata = {k: v for k, v in row.items() if k not in {id_column, *text_columns} | ignored_annotations}
        result.append(Document(row[id_column], "\n".join(pieces), metadata))
    return result


def tokenize_and_chunk(documents: Iterable[Document], tokenizer, max_length: int, stride: int = 0):
    """Tokenize each document separately; offsets remain raw-document offsets."""
    if max_length < 2 or stride < 0 or stride >= max_length:
        raise ValueError("Require max_length >= 2 and 0 <= stride < max_length")
    sequences, stats = [], CorpusStats()
    step = max_length - stride
    for doc in documents:
        encoded = tokenizer(doc.text, add_special_tokens=False, return_offsets_mapping=True)
        ids, offsets = encoded["input_ids"], encoded["offset_mapping"]
        stats.documents += 1; stats.characters += len(doc.text); stats.tokens += len(ids)
        for start in range(0, len(ids), step):
            chunk_ids, chunk_offsets = ids[start:start + max_length], offsets[start:start + max_length]
            if not chunk_ids: break
            prepared = tokenizer.prepare_for_model(chunk_ids, add_special_tokens=True, return_special_tokens_mask=True)
            n_special_left = next((i for i, x in enumerate(prepared["special_tokens_mask"]) if not x), len(prepared["input_ids"]))
            prepared["offset_mapping"] = [(0, 0)] * n_special_left + list(chunk_offsets)
            prepared["offset_mapping"] += [(0, 0)] * (len(prepared["input_ids"]) - len(prepared["offset_mapping"]))
            prepared.update(doc_id=doc.doc_id, text=doc.text, metadata=dict(doc.metadata))
            sequences.append(prepared)
            if start + max_length >= len(ids): break
    stats.sequences = len(sequences)
    return sequences, stats
