"""Target-normalized optimization with epoch-boundary resume and independent evaluation."""

from __future__ import annotations

import json
import math
import random
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import torch
from accelerate import Accelerator
from accelerate.utils import set_seed
from torch.nn import functional as functional
from transformers import get_scheduler

from lab.cpt.contracts import Window
from lab.cpt.evaluation.intrinsic import evaluate_pretraining, pseudo_likelihood
from lab.cpt.importance.uncertainty import predictive_entropy
from lab.cpt.masking.collator import MaskCollator
from lab.cpt.masking.policies import MaskPolicy, WindowSignals
from lab.cpt.modeling import bidirectional_attention, checkpoint_sha256
from lab.cpt.training.arguments import TrainingParameters
from lab.cpt.training.checkpoints import save_checkpoint
from lab.cpt.training.tracking import RunTracker, write_json


@dataclass(frozen=True)
class PretrainingResult:
    """Persisted results without retaining a model or accelerator in memory."""

    output_dir: Path
    final_checkpoint: Path
    final_sha256: str
    best_checkpoint: Path
    global_step: int


def train(
    *,
    model: Any,
    tokenizer: Any,
    training_windows: Sequence[Window],
    validation_windows: Sequence[Window],
    output_dir: Path,
    objective: str,
    parameters: TrainingParameters,
    policy: MaskPolicy,
    signals: Mapping[tuple[str, int], WindowSignals],
    manifest: dict[str, Any],
    resume_from: str | Path | None = None,
    stop_after_epoch: int | None = None,
    evaluation_probe: Callable[..., dict[str, float]] | None = None,
    selection_metric: str = "nll",
    greater_is_better: bool = False,
) -> PretrainingResult:
    """Train single-process CPU/GPU; accumulation normalizes by all group targets.

    Resume is explicitly at epoch boundaries with an unchanged finite training plan.
    Distributed launches fail early until their target aggregation is verified.
    External probes receive a bidirectional model and validation windows, never final test.
    """
    accelerator = Accelerator(mixed_precision=parameters.mixed_precision)
    if accelerator.num_processes != 1:
        raise ValueError("Distributed CPT is not verified; launch one process.")
    if not training_windows or not validation_windows:
        raise ValueError("Training and separate validation windows must both be nonempty.")
    if selection_metric != "nll" and evaluation_probe is None:
        raise ValueError("A downstream selection metric requires a validation evaluation_probe.")
    if stop_after_epoch is not None and not 1 <= stop_after_epoch <= parameters.epochs:
        raise ValueError("stop_after_epoch must be within the declared training plan.")
    set_seed(parameters.training_seed)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=parameters.learning_rate, weight_decay=parameters.weight_decay
    )
    batches = math.ceil(len(training_windows) / parameters.batch_size)
    total_steps = math.ceil(batches / parameters.gradient_accumulation) * parameters.epochs
    scheduler = get_scheduler(
        parameters.scheduler,
        optimizer=optimizer,
        num_warmup_steps=int(total_steps * parameters.warmup_ratio),
        num_training_steps=total_steps,
    )
    model, optimizer, scheduler = accelerator.prepare(model, optimizer, scheduler)
    tracker = RunTracker(output_dir=output_dir)
    state: dict[str, Any] = {
        "epoch": 0,
        "global_step": 0,
        "processed_tokens": 0,
        "best_value": None,
        "best_epoch": 0,
        "best_checkpoint": None,
        "plan": manifest["parameters"],
        "mask_epoch": 0,
    }
    if resume_from:
        saved = json.loads((Path(resume_from) / "training_state.json").read_text())
        if saved["plan"] != json.loads(json.dumps(manifest["parameters"])):
            raise ValueError("Resume plan/corpus/parent checkpoint differs; use the original plan.")
        accelerator.load_state(str(Path(resume_from) / "accelerate"))
        state = saved
        tracker.rows = saved["metric_rows"]
    collator = MaskCollator(
        tokenizer=tokenizer, policy=policy, signals=signals, objective=objective
    )
    uncertainty_cache = {
        (item["doc_id"], item["window_index"]): tuple(item["scores"])
        for item in state.get("uncertainty_cache", [])
    }
    writer = None
    if parameters.tensorboard:
        from torch.utils.tensorboard import SummaryWriter

        writer = SummaryWriter(log_dir=str(output_dir / "tensorboard"))
    try:
        if not resume_from:
            _evaluate(
                model,
                tokenizer,
                validation_windows,
                parameters,
                objective,
                tracker,
                state,
                evaluation_probe,
                selection_metric,
                greater_is_better,
            )
            save_checkpoint(
                accelerator=accelerator,
                model=model,
                tokenizer=tokenizer,
                path=output_dir / "checkpoints" / "epoch_000",
                state={**state, "metric_rows": tracker.rows},
            )
        for epoch in range(state["epoch"] + 1, (stop_after_epoch or parameters.epochs) + 1):
            collator.epoch = epoch
            extra_seconds = 0.0
            if policy.policy == "M8" and (
                not uncertainty_cache or (epoch - 1) % policy.adaptive_refresh_epochs == 0
            ):
                uncertainty_cache, extra_seconds = predictive_entropy(
                    model=accelerator.unwrap_model(model),
                    tokenizer=tokenizer,
                    windows=training_windows,
                )
                collator.signals = {
                    key: replace(signals.get(key, WindowSignals()), uncertainty=value)
                    for key, value in uncertainty_cache.items()
                }
            if policy.policy == "M8":
                collator.signals = {
                    key: replace(signals.get(key, WindowSignals()), uncertainty=value)
                    for key, value in uncertainty_cache.items()
                }
                state["uncertainty_cache"] = [
                    {"doc_id": key[0], "window_index": key[1], "scores": value}
                    for key, value in uncertainty_cache.items()
                ]
            metrics = _train_epoch(
                model,
                optimizer,
                scheduler,
                accelerator,
                training_windows,
                collator,
                parameters,
                epoch,
                state,
            )
            metrics["adaptive_inference_seconds"] = extra_seconds
            tracker.record(
                epoch=epoch, step=state["global_step"], group="training", metrics=metrics
            )
            if epoch % parameters.evaluation_every_epochs == 0 or epoch == parameters.epochs:
                _evaluate(
                    model,
                    tokenizer,
                    validation_windows,
                    parameters,
                    objective,
                    tracker,
                    state,
                    evaluation_probe,
                    selection_metric,
                    greater_is_better,
                )
            state.update(epoch=epoch, mask_epoch=epoch)
            save_checkpoint(
                accelerator=accelerator,
                model=model,
                tokenizer=tokenizer,
                path=output_dir / "checkpoints" / f"epoch_{epoch:03d}",
                state={**state, "metric_rows": tracker.rows},
            )
            if writer:
                for name, value in metrics.items():
                    writer.add_scalar(f"training/{name}", value, state["global_step"])
        final = output_dir / "checkpoints" / f"epoch_{state['epoch']:03d}" / "model"
        result = PretrainingResult(
            output_dir,
            final,
            checkpoint_sha256(model_dir=final),
            Path(state["best_checkpoint"]),
            state["global_step"],
        )
        write_json(
            path=output_dir / "training_summary.json",
            content={
                "global_step": result.global_step,
                "final_checkpoint": str(result.final_checkpoint),
                "final_sha256": result.final_sha256,
                "best_checkpoint": str(result.best_checkpoint),
                "selection_metric": selection_metric,
                "best_value": state["best_value"],
                "resume_boundary": "epoch",
                "completed_epochs": state["epoch"],
            },
        )

        return result
    finally:
        if writer:
            writer.close()
        accelerator.end_training()
        accelerator.free_memory()


def _train_epoch(
    model: Any,
    optimizer: Any,
    scheduler: Any,
    accelerator: Accelerator,
    windows: Sequence[Window],
    collator: MaskCollator,
    parameters: TrainingParameters,
    epoch: int,
    state: dict[str, Any],
) -> dict[str, float | int]:
    started = time.perf_counter()
    order = list(range(len(windows)))
    random.Random(parameters.data_seed + epoch).shuffle(order)
    group_size = parameters.batch_size * parameters.gradient_accumulation
    collator.metrics.clear()
    collator.word_types.clear()
    collator.concepts.clear()
    model.train()
    nll = targets = tokens = updates = correct = 0
    gradient_norm = 0.0
    for start in range(0, len(order), group_size):
        group = [windows[index] for index in order[start : start + group_size]]
        batches = [
            collator(group[index : index + parameters.batch_size])
            for index in range(0, len(group), parameters.batch_size)
        ]
        group_targets = sum(int((batch["labels"] != -100).sum()) for batch in batches)
        if not group_targets:
            raise ValueError("Optimization group has zero targets; increase masking probability.")
        optimizer.zero_grad(set_to_none=True)
        for batch in batches:
            batch = {key: value.to(accelerator.device) for key, value in batch.items()}
            labels = batch.pop("labels")
            with accelerator.autocast():
                logits = model(**batch).logits.float()
                loss_sum = functional.cross_entropy(
                    logits.transpose(1, 2), labels, ignore_index=-100, reduction="sum"
                )
            if not torch.isfinite(loss_sum):
                raise ValueError("Training loss is nonfinite; inspect model and numeric precision.")
            accelerator.backward(loss_sum / group_targets)
            correct += int(((logits.detach().argmax(-1) == labels) & (labels != -100)).sum())
            nll += float(loss_sum.detach())
            targets += int((labels != -100).sum())
            tokens += int(batch["attention_mask"].sum())
        gradient_norm += float(
            accelerator.clip_grad_norm_(model.parameters(), parameters.max_grad_norm)
        )
        optimizer.step()
        if not accelerator.optimizer_step_was_skipped:
            scheduler.step()
            updates += 1
            state["global_step"] += 1
    state["processed_tokens"] += tokens
    elapsed = time.perf_counter() - started
    metrics = {
        "loss": nll / targets,
        "prediction_accuracy": correct / targets,
        "prediction_targets": targets,
        "processed_tokens": tokens,
        "learning_rate": scheduler.get_last_lr()[0],
        "gradient_norm": gradient_norm / max(1, updates),
        "optimizer_steps": updates,
        "wall_seconds": elapsed,
        "tokens_per_second": tokens / max(elapsed, 1e-9),
        "effective_batch_size": group_size,
        "gpu_memory_bytes": torch.cuda.max_memory_allocated() if torch.cuda.is_available() else 0,
        "gpu_hours": elapsed / 3600 if torch.cuda.is_available() else 0,
    }
    totals = {
        key: sum(row.get(key, 0) for row in collator.metrics)
        for key in {key for row in collator.metrics for key in row}
    }
    totals.pop("unique_word_types", None)
    totals.pop("unique_concepts", None)
    metrics.update(totals)
    metrics.update(
        unique_word_types=len(collator.word_types),
        unique_concepts=len(collator.concepts),
        effective_masking_rate=totals.get("selected_targets", 0)
        / max(1, totals.get("eligible_subwords", 0)),
        entity_target_share=totals.get("entity_targets", 0)
        / max(1, totals.get("selected_targets", 0)),
        random_backfill_fraction=totals.get("backfill_targets", 0)
        / max(1, totals.get("selected_targets", 0)),
        mean_selected_span_length=totals.get("total_span_length", 0)
        / max(1, totals.get("selected_units", 0)),
    )

    return metrics


def _evaluate(
    model: Any,
    tokenizer: Any,
    windows: Sequence[Window],
    parameters: TrainingParameters,
    objective: str,
    tracker: RunTracker,
    state: dict[str, Any],
    probe: Callable[..., dict[str, float]] | None,
    selection_metric: str,
    greater_is_better: bool,
) -> None:
    epoch = state["epoch"] if state["global_step"] == 0 else state["epoch"] + 1
    metrics = evaluate_pretraining(
        model=model,
        tokenizer=tokenizer,
        windows=windows,
        objective=objective,
        probability=parameters.evaluation_probability,
        evaluation_seed=parameters.evaluation_seed,
        batch_size=parameters.batch_size,
    )
    if parameters.evaluate_pll:
        metrics.update(
            {
                f"word_{name}": value
                for name, value in pseudo_likelihood(
                    model=model, tokenizer=tokenizer, windows=windows
                ).items()
            }
        )
    if probe:
        with bidirectional_attention(model=model):
            metrics.update(probe(model=model, tokenizer=tokenizer, windows=windows, epoch=epoch))
    if selection_metric not in metrics or not math.isfinite(metrics[selection_metric]):
        raise ValueError(
            f"Validation did not produce finite selection_metric={selection_metric!r}."
        )
    value = metrics[selection_metric]
    if state["best_value"] is None or (
        value > state["best_value"] if greater_is_better else value < state["best_value"]
    ):
        state.update(
            best_value=value,
            best_epoch=epoch,
            best_checkpoint=str(
                tracker.output_dir / "checkpoints" / f"epoch_{epoch:03d}" / "model"
            ),
        )
    tracker.record(epoch=epoch, step=state["global_step"], group="validation", metrics=metrics)
    evaluation_dir = tracker.output_dir / "evaluations"
    evaluation_dir.mkdir(exist_ok=True)
    write_json(path=evaluation_dir / f"epoch_{epoch:03d}.json", content=metrics)
