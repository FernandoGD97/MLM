"""Real Transformer incoming-attention scores projected to complete word occurrences."""

from __future__ import annotations

import time
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from lab.cpt.inspection.corpus import Document


@dataclass(frozen=True)
class AttentionScores:
    """Occurrence-aligned salience and the measured model/aggregation provenance."""

    scores: tuple[tuple[float, ...], ...]
    checkpoint_sha256: str
    seconds: float
    layers: int
    max_length: int


def incoming_attention(
    *,
    attentions: Sequence[Any],
    eligible: Sequence[int],
    word_positions: Sequence[Sequence[int]],
    layers: int = 1,
) -> tuple[float, ...]:
    """Average received attention over last layers, heads and nonspecial query tokens.

    Rescale eligible-key scores to mean one, then average a word's subwords. This avoids
    automatically rewarding fragmented words and makes uniform attention score one.
    Tensor layout is (batch=1, heads, queries, keys). No special/padding positions vote.
    """
    import torch

    if layers < 1 or layers > len(attentions) or not eligible:
        raise ValueError("Supply eligible positions and 1 <= attention_layers <= model layers.")
    selected = []
    for attention in attentions[-layers:]:
        if attention is None or attention.ndim != 4 or attention.shape[0] != 1:
            raise ValueError("Encoder must return real attention tensors with batch size one.")
        if (
            attention.shape[-1] != attention.shape[-2]
            or len(set(eligible)) != len(eligible)
            or any(position < 0 or position >= attention.shape[-1] for position in eligible)
        ):
            raise ValueError(
                "Eligible positions must index distinct tokens in square self-attention."
            )
        if not torch.isfinite(attention).all() or (attention < 0).any():
            raise ValueError("Attention probabilities must be finite and nonnegative.")
        matrix = attention[0].float()[:, list(eligible), :][:, :, list(eligible)]
        selected.append(matrix.mean(dim=(0, 1)))
    received = torch.stack(selected).mean(0)
    if float(received.mean()) <= 0:
        raise ValueError("No attention reaches eligible content; cannot rank its words.")
    received /= received.mean()
    indices = {position: index for index, position in enumerate(eligible)}
    if any(
        not positions or any(position not in indices for position in positions)
        for positions in word_positions
    ):
        raise ValueError("Word subwords must be nonempty subsets of eligible positions.")

    return tuple(
        float(received[[indices[position] for position in positions]].mean())
        for positions in word_positions
    )


def attention_signals(
    *,
    documents: Sequence[Document],
    base_model: str,
    layers: int = 1,
    max_length: int = 128,
) -> AttentionScores:
    """Run an explicit local encoder in eager/eval mode, with bounded document windows.

    Attention is an experimental salience heuristic, not proof of clinical importance.
    Models without reliable offsets or returned attention probabilities fail explicitly.
    """
    import pandas as pd
    import torch
    from transformers import AutoConfig, AutoModel, AutoTokenizer

    from lab.cpt.corpus import tokenize_corpus
    from lab.cpt.modeling import checkpoint_sha256

    fingerprint = checkpoint_sha256(model_dir=base_model)
    started = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(base_model, local_files_only=True)
    config = AutoConfig.from_pretrained(base_model, local_files_only=True)
    options = {"add_pooling_layer": False} if config.model_type in {"bert", "roberta"} else {}
    model, loading = AutoModel.from_pretrained(
        base_model,
        local_files_only=True,
        attn_implementation="eager",
        output_loading_info=True,
        **options,
    )
    if loading.get("missing_keys") or loading.get("mismatched_keys"):
        raise ValueError("Checkpoint lacks compatible encoder weights; supply a complete backbone.")
    if getattr(model.config, "is_decoder", False):
        raise ValueError("Attention exploration requires a bidirectional encoder checkpoint.")
    if layers < 1 or layers > getattr(model.config, "num_hidden_layers", 0):
        raise ValueError("attention_layers must be between one and the encoder layer count.")
    model.eval()
    frame = pd.DataFrame(
        [{"doc_id": str(index), "text": doc.text} for index, doc in enumerate(documents)]
    )
    windows = tokenize_corpus(documents=frame, tokenizer=tokenizer, max_length=max_length)
    lookup = {}
    with torch.inference_mode():
        for window in windows:
            if not window.eligible:
                continue
            ids = torch.tensor([window.input_ids])
            result = model(
                input_ids=ids, attention_mask=torch.ones_like(ids), output_attentions=True
            )
            if not result.attentions:
                raise ValueError(
                    "This encoder does not expose attention probabilities in eager mode."
                )
            scores = incoming_attention(
                attentions=result.attentions,
                eligible=window.eligible,
                word_positions=[word.positions for word in window.words],
                layers=layers,
            )
            for word, score in zip(window.words, scores, strict=True):
                lookup[(int(window.doc_id), word.start, word.end)] = score
    aligned = tuple(
        tuple(lookup[(index, start, end)] for start, end in doc.words)
        for index, doc in enumerate(documents)
    )

    return AttentionScores(aligned, fingerprint, time.perf_counter() - started, layers, max_length)
