"""Explicit comparable-run validation and tidy summaries from persisted artifacts."""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path

import pandas as pd

CONTROLLED_FIELDS = (
    "parent_sha256",
    "training_corpus_sha256",
    "validation_corpus_sha256",
    "max_length",
    "training",
    "selection_metric",
    "greater_is_better",
    "evaluation_probe",
)
CONTROLLED_MASK_FIELDS = ("probability", "mask_seed", "mask_fraction", "random_fraction")


def compare_pretraining_runs(
    *,
    runs: Sequence[str | Path],
    output_dir: str | Path | None = None,
    require_comparable: bool = True,
) -> pd.DataFrame:
    """Compare completed MLM branches only after checking parent, seeds and budgets.

    Underfilled policies are rejected in strict comparison mode. Set require_comparable
    false explicitly to inspect exploratory runs without asserting controlled comparability.
    """
    rows = []
    manifests = [json.loads((Path(run) / "run_manifest.json").read_text()) for run in runs]
    if not manifests:
        raise ValueError("Supply at least one run.")
    baseline = manifests[0]["parameters"]
    for run, manifest in zip(runs, manifests, strict=True):
        plan = manifest["parameters"]
        if manifest["status"] != "completed":
            raise ValueError(f"Run {run} is not completed.")
        differences = [name for name in CONTROLLED_FIELDS if plan[name] != baseline[name]]
        differences += [
            f"mask_policy.{name}"
            for name in CONTROLLED_MASK_FIELDS
            if plan["mask_policy"][name] != baseline["mask_policy"][name]
        ]
        if require_comparable and (differences or plan["objective"] != "mlm"):
            raise ValueError(
                f"Runs are not controlled MLM branches; differing fields: {differences}."
            )
        metrics = pd.read_parquet(Path(run) / "epoch_metrics.parquet")
        deviation = metrics[metrics.metric == "budget_deviation"].value.abs().sum()
        if require_comparable and deviation:
            raise ValueError(
                f"{run} has masking-budget deviation {deviation}; "
                "compare explicitly as exploratory."
            )
        summary = json.loads((Path(run) / "training_summary.json").read_text())
        rows.append(
            {
                "run": str(run),
                "policy": plan["mask_policy"]["policy"],
                "parent_sha256": plan["parent_sha256"],
                "training_seed": plan["training"]["training_seed"],
                "mask_seed": plan["mask_policy"]["mask_seed"],
                "steps": summary["global_step"],
                "best_value": summary["best_value"],
                "selection_metric": summary["selection_metric"],
                "budget_deviation": deviation,
                "wall_seconds": manifest["wall_seconds"],
                "gpu_hours": metrics[metrics.metric == "gpu_hours"].value.sum(),
                "prediction_targets": metrics[
                    (metrics.metric == "prediction_targets") & (metrics.group == "training")
                ].value.sum(),
                "adaptive_inference_seconds": metrics[
                    metrics.metric == "adaptive_inference_seconds"
                ].value.sum(),
                "controlled": require_comparable,
            }
        )
    table = pd.DataFrame(rows)
    if output_dir is not None:
        output = Path(output_dir)
        output.mkdir(parents=True, exist_ok=False)
        table.to_parquet(output / "comparison.parquet", index=False)
        table.to_csv(output / "comparison.tsv", sep="\t", index=False)

    return table
