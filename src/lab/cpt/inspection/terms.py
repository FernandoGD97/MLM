"""Explicit key-term readers for extractor-produced TXT and single-column Parquet lists."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import pandas as pd


def read_terms(*, paths: Sequence[str | Path]) -> tuple[str, ...]:
    """Read literal words or phrases, removing blank TXT lines and exact duplicates.

    Parquet must contain exactly one column named terms, with nonempty string cells.
    Nulls and non-string cells fail instead of becoming accidental masking terms.
    """
    if isinstance(paths, (str, Path)):
        raise ValueError("paths must be a list of TXT/Parquet files.")
    result = []
    for path in paths:
        source = Path(path)
        if source.suffix.lower() == ".txt":
            values = [
                line.strip()
                for line in source.read_text(encoding="utf-8-sig").splitlines()
                if line.strip()
            ]
        elif source.suffix.lower() == ".parquet":
            frame = pd.read_parquet(source)
            if list(frame.columns) != ["terms"]:
                raise ValueError(f"{source}: expected a single Parquet column named 'terms'.")
            values = frame.terms.tolist()
        else:
            raise ValueError(f"{source}: term sources must be .txt or .parquet files.")
        if not values:
            raise ValueError(f"{source}: the term list is empty.")
        if any(not isinstance(value, str) or not value.strip() for value in values):
            raise ValueError(f"{source}: terms must be nonempty strings; remove null/invalid rows.")
        result.extend(value.strip() for value in values)

    return tuple(dict.fromkeys(result))
