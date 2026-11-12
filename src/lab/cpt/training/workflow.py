"""Public Python/YAML entrypoints with explicit validation, provenance and run ownership."""

from __future__ import annotations

import json
import math
import time
from collections.abc import Callable, Mapping
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

import pandas as pd
import yaml
from transformers import AutoModelForMaskedLM, AutoTokenizer

from lab.cpt.corpus import corpus_fingerprint, prepare_corpus, tokenize_corpus
from lab.cpt.importance.domain import score_domain_words
from lab.cpt.importance.entities import read_entity_spans
from lab.cpt.importance.preparation import prepare_signals
from lab.cpt.masking.policies import MaskPolicy
from lab.cpt.modeling import checkpoint_sha256, set_attention_mode, verify_attention
from lab.cpt.sampling import describe_corpus
from lab.cpt.training.arguments import TrainingParameters
from lab.cpt.training.tracking import initial_manifest, record_status, write_json
from lab.cpt.training.trainer import PretrainingResult, train


def pretrain_causal(
    *,
    base_model: str,
    documents: pd.DataFrame | str | Path,
    validation: pd.DataFrame | str | Path,
    output_dir: str | Path,
    training_parameters: TrainingParameters | Mapping[str, Any] | None = None,
    max_length: int = 128,
    random_state: int = 42,
    resume_from: str | Path | None = None,
    stop_after_epoch: int | None = None,
) -> PretrainingResult:
    """Continue a supported encoder causally, saving full shared checkpoints.

    No test data are read. The mapping form has exactly TrainingParameters' schema.
    Each invocation owns a new output directory, including resumed runs.
    """
    parameters = _parameters(training_parameters, random_state)

    return _pretrain(
        base_model=base_model,
        documents=documents,
        validation=validation,
        output_dir=output_dir,
        parameters=parameters,
        policy=MaskPolicy(),
        objective="clm",
        max_length=max_length,
        resume_from=resume_from,
        stop_after_epoch=stop_after_epoch,
    )


def pretrain_masked(
    *,
    base_model: str,
    documents: pd.DataFrame | str | Path,
    validation: pd.DataFrame | str | Path,
    output_dir: str | Path,
    training_parameters: TrainingParameters | Mapping[str, Any] | None = None,
    mask_policy: MaskPolicy | Mapping[str, Any] | None = None,
    max_length: int = 128,
    random_state: int = 42,
    entity_predictions: pd.DataFrame | str | Path | None = None,
    entity_mode: str = "binary",
    entity_type_weights: Mapping[str, float] | None = None,
    domain_method: str = "tfidf",
    reference_documents: pd.DataFrame | str | Path | None = None,
    minimum_frequency: int = 2,
    parent_sha256: str | None = None,
    resume_from: str | Path | None = None,
    stop_after_epoch: int | None = None,
    evaluation_probe: Callable[..., dict[str, float]] | None = None,
    selection_metric: str = "nll",
    greater_is_better: bool = False,
) -> PretrainingResult:
    """Train one MLM branch from a content-addressed common CLM or original checkpoint.

    Supply the same local base_model and parent_sha256 for all CLM→MLM branches.
    Predictions are separate from corpus gold fields. Evaluation always uses fixed random
    masks independent of the selected policy; downstream probes are composed externally.
    """
    policy = (
        mask_policy if isinstance(mask_policy, MaskPolicy) else MaskPolicy(**(mask_policy or {}))
    )
    parameters = _parameters(training_parameters, random_state)

    return _pretrain(
        base_model=base_model,
        documents=documents,
        validation=validation,
        output_dir=output_dir,
        parameters=parameters,
        policy=policy,
        objective="mlm",
        max_length=max_length,
        resume_from=resume_from,
        stop_after_epoch=stop_after_epoch,
        entity_predictions=entity_predictions,
        entity_mode=entity_mode,
        entity_type_weights=entity_type_weights,
        domain_method=domain_method,
        reference_documents=reference_documents,
        minimum_frequency=minimum_frequency,
        parent_sha256=parent_sha256,
        evaluation_probe=evaluation_probe,
        selection_metric=selection_metric,
        greater_is_better=greater_is_better,
    )


def _parameters(
    value: TrainingParameters | Mapping[str, Any] | None, random_state: int
) -> TrainingParameters:
    parameters = (
        value if isinstance(value, TrainingParameters) else TrainingParameters(**(value or {}))
    )

    return replace(parameters, training_seed=random_state)


def _pretrain(
    *,
    base_model: str,
    documents: Any,
    validation: Any,
    output_dir: str | Path,
    parameters: TrainingParameters,
    policy: MaskPolicy,
    objective: str,
    max_length: int,
    resume_from: str | Path | None,
    stop_after_epoch: int | None,
    entity_predictions: Any = None,
    entity_mode: str = "binary",
    entity_type_weights: Mapping[str, float] | None = None,
    domain_method: str = "tfidf",
    reference_documents: Any = None,
    minimum_frequency: int = 2,
    parent_sha256: str | None = None,
    evaluation_probe: Callable[..., dict[str, float]] | None = None,
    selection_metric: str = "nll",
    greater_is_better: bool = False,
) -> PretrainingResult:
    output = Path(output_dir)
    if output.exists():
        raise ValueError(f"Run directory already exists: {output}; choose a new output_dir.")
    if not Path(base_model).is_dir():
        raise ValueError(
            "base_model must be an explicit local saved checkpoint; download/revise it separately."
        )
    parent = checkpoint_sha256(model_dir=base_model)
    if parent_sha256 is not None and parent != parent_sha256:
        raise ValueError(
            "Parent checkpoint hash differs from parent_sha256; restore the common M_C."
        )
    training_df = prepare_corpus(documents=documents)
    validation_df = prepare_corpus(documents=validation)
    _separate_corpora(training_df, validation_df)
    tokenizer = AutoTokenizer.from_pretrained(base_model, use_fast=True, local_files_only=True)
    training_windows = tokenize_corpus(
        documents=training_df, tokenizer=tokenizer, max_length=max_length
    )
    validation_windows = tokenize_corpus(
        documents=validation_df, tokenizer=tokenizer, max_length=max_length
    )
    predictions = (
        read_entity_spans(path=entity_predictions)
        if isinstance(entity_predictions, (str, Path))
        else entity_predictions
    )
    domain_scores = None
    if policy.policy in {"M3", "M4", "M5", "M6", "M7"}:
        reference_df = (
            prepare_corpus(documents=reference_documents)
            if reference_documents is not None
            else None
        )
        domain_scores = score_domain_words(
            training_texts=training_df.text.tolist(),
            reference_texts=reference_df.text.tolist() if reference_df is not None else (),
            method=domain_method,
            minimum_frequency=minimum_frequency,
        )
    signals = prepare_signals(
        documents=training_df,
        windows=training_windows,
        policy=policy,
        entity_spans=predictions,
        domain_scores=domain_scores,
        entity_mode=entity_mode,
        type_weights=entity_type_weights,
    )
    model = AutoModelForMaskedLM.from_pretrained(
        base_model, local_files_only=True, attn_implementation="eager"
    )
    set_attention_mode(model=model, objective=objective)
    verify_attention(model=model, tokenizer=tokenizer, objective=objective)
    plan = {
        "base_model": str(Path(base_model).resolve()),
        "parent_sha256": parent,
        "objective": objective,
        "max_length": max_length,
        "training": asdict(parameters),
        "mask_policy": asdict(policy),
        "training_corpus_sha256": corpus_fingerprint(documents=training_df),
        "validation_corpus_sha256": corpus_fingerprint(documents=validation_df),
        "training_metadata_columns": list(training_df.columns),
        "domain_scores": {
            "method": domain_scores.method,
            "training_sha256": domain_scores.training_sha256,
            "minimum_frequency": domain_scores.minimum_frequency,
            "smoothing": domain_scores.smoothing,
            "vocabulary_size": len(domain_scores.scores),
        }
        if domain_scores
        else None,
        "entity_predictions_sha256": corpus_fingerprint(documents=predictions)
        if predictions is not None
        else None,
        "entity_mode": entity_mode,
        "entity_type_weights": dict(entity_type_weights or {}),
        "selection_metric": selection_metric,
        "greater_is_better": greater_is_better,
        "evaluation_probe": f"{evaluation_probe.__module__}.{evaluation_probe.__qualname__}"
        if evaluation_probe
        else None,
    }
    manifest = initial_manifest(parameters=plan)
    manifest["corpus_statistics"] = {
        "training": describe_corpus(documents=training_df),
        "validation": describe_corpus(documents=validation_df),
    }
    manifest["model_architecture"] = model.config.model_type
    manifest["model_class"] = type(model).__name__
    manifest["target_optimizer_steps"] = (
        math.ceil(
            len(training_windows) / (parameters.batch_size * parameters.gradient_accumulation)
        )
        * parameters.epochs
    )
    manifest["target_processed_tokens"] = (
        sum(len(window.input_ids) for window in training_windows) * parameters.epochs
    )
    output.mkdir(parents=True)
    (output / "checkpoints").mkdir()
    (output / "resolved_config.yaml").write_text(
        yaml.safe_dump(json.loads(json.dumps(plan)), sort_keys=True)
    )
    write_json(path=output / "run_manifest.json", content=manifest)
    record_status(output_dir=output, status="running", run_id=manifest["run_id"])
    started = time.perf_counter()
    try:
        result = train(
            model=model,
            tokenizer=tokenizer,
            training_windows=training_windows,
            validation_windows=validation_windows,
            output_dir=output,
            objective=objective,
            parameters=parameters,
            policy=policy,
            signals=signals,
            manifest=manifest,
            resume_from=resume_from,
            stop_after_epoch=stop_after_epoch,
            evaluation_probe=evaluation_probe,
            selection_metric=selection_metric,
            greater_is_better=greater_is_better,
        )
        if checkpoint_sha256(model_dir=base_model) != parent:
            raise ValueError(
                "Parent checkpoint changed during training; invalidate this comparison."
            )
        manifest.update(
            status="completed"
            if not stop_after_epoch or stop_after_epoch == parameters.epochs
            else "paused",
            final_sha256=result.final_sha256,
            final_checkpoint=str(result.final_checkpoint),
        )

        return result
    except BaseException as error:
        manifest.update(status="failed", error_type=type(error).__name__)
        raise
    finally:
        record_status(
            output_dir=output,
            status=manifest["status"],
            run_id=manifest["run_id"],
            error_type=manifest.get("error_type"),
        )
        manifest["wall_seconds"] = time.perf_counter() - started
        write_json(path=output / "run_manifest.json", content=manifest)


def _separate_corpora(training: pd.DataFrame, validation: pd.DataFrame) -> None:
    def families(frame: pd.DataFrame) -> set[str]:
        values = set(frame.doc_id)
        if "original_doc_id" in frame:
            values.update(
                value for value in frame.original_doc_id if isinstance(value, str) and value
            )
        return values

    if families(training) & families(validation):
        raise ValueError("Train/dev document families overlap; use grouped_split before training.")
    if set(training.text) & set(validation.text):
        raise ValueError("Train/dev contain duplicate text; deduplicate before splitting.")
