"""Explicit original, intermediate, best and final checkpoint evaluation artifacts."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from transformers import AutoModelForMaskedLM, AutoTokenizer

from lab.cpt.corpus import corpus_fingerprint, prepare_corpus, tokenize_corpus
from lab.cpt.evaluation.intrinsic import evaluate_pretraining, pseudo_likelihood
from lab.cpt.modeling import checkpoint_sha256, set_attention_mode, verify_attention
from lab.cpt.training.tracking import write_json


@dataclass(frozen=True)
class CheckpointEvaluation:
    """Metrics and the immutable identities of the evaluated model and corpus."""

    checkpoint_sha256: str
    corpus_sha256: str
    metrics: tuple[tuple[str, float | int], ...]


def evaluate_checkpoint(
    *,
    base_model: str,
    documents: pd.DataFrame | str | Path,
    objective: str = "mlm",
    max_length: int = 128,
    probability: float = 0.15,
    evaluation_seed: int = 44,
    evaluate_pll: bool = False,
    output_dir: str | Path | None = None,
) -> CheckpointEvaluation:
    """Evaluate any saved encoder with the same fixed masks and conditioning.

    Disk output is optional and never overwrites an existing artifact directory.
    Use explicitly supplied dev documents for model selection, final test only afterward.
    """
    if output_dir is not None and Path(output_dir).exists():
        raise ValueError("Evaluation output_dir already exists; choose a fresh directory.")
    frame = prepare_corpus(documents=documents)
    tokenizer = AutoTokenizer.from_pretrained(base_model, local_files_only=True, use_fast=True)
    model = AutoModelForMaskedLM.from_pretrained(
        base_model, local_files_only=True, attn_implementation="eager"
    )
    set_attention_mode(model=model, objective=objective)
    verify_attention(model=model, tokenizer=tokenizer, objective=objective)
    windows = tokenize_corpus(documents=frame, tokenizer=tokenizer, max_length=max_length)
    metrics = evaluate_pretraining(
        model=model,
        tokenizer=tokenizer,
        windows=windows,
        objective=objective,
        probability=probability,
        evaluation_seed=evaluation_seed,
    )
    if evaluate_pll:
        metrics.update(
            {
                f"word_{key}": value
                for key, value in pseudo_likelihood(
                    model=model, tokenizer=tokenizer, windows=windows
                ).items()
            }
        )
    result = CheckpointEvaluation(
        checkpoint_sha256(model_dir=base_model),
        corpus_fingerprint(documents=frame),
        tuple(sorted(metrics.items())),
    )
    if output_dir is not None:
        output = Path(output_dir)
        output.mkdir(parents=True, exist_ok=False)
        write_json(
            path=output / "metrics.json",
            content={
                "checkpoint_sha256": result.checkpoint_sha256,
                "corpus_sha256": result.corpus_sha256,
                "objective": objective,
                "evaluation_seed": evaluation_seed,
                "probability": probability,
                "max_length": max_length,
                "metrics": metrics,
            },
        )

    return result
