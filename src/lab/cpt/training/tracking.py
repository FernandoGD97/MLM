"""Explicit atomic manifests, structured events and tidy epoch metric persistence."""

from __future__ import annotations

import importlib.metadata
import json
import platform
import subprocess
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import torch


def write_json(*, path: Path, content: Any) -> None:
    """Replace a JSON artifact atomically and reject nonfinite metrics."""
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(content, indent=2, sort_keys=True, allow_nan=False), encoding="utf-8"
    )
    temporary.replace(path)


def initial_manifest(*, parameters: dict[str, Any]) -> dict[str, Any]:
    """Record installed code location, safe hardware metadata and complete experiment inputs."""
    root = Path(__file__).resolve().parents[4]
    commit = None
    dirty = None
    try:
        commit = subprocess.check_output(
            ["git", "-C", str(root), "rev-parse", "HEAD"], text=True
        ).strip()
        dirty = bool(
            subprocess.check_output(
                ["git", "-C", str(root), "status", "--porcelain"], text=True
            ).strip()
        )
    except (OSError, subprocess.CalledProcessError):
        pass

    return {
        "format_version": 1,
        "run_id": uuid.uuid4().hex,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "running",
        "git_commit": commit,
        "git_dirty": dirty,
        "parameters": parameters,
        "hardware": {
            "platform": platform.platform(),
            "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
        },
        "versions": {
            name: importlib.metadata.version(name)
            for name in ("lab", "torch", "transformers", "accelerate", "pyarrow")
        },
    }


class RunTracker:
    """Retain numeric rows and explicitly persist aggregate metrics and events."""

    def __init__(self, *, output_dir: Path, rows: list[dict[str, Any]] | None = None) -> None:
        self.output_dir = output_dir
        self.rows = rows or []

    def record(self, *, epoch: int, step: int, group: str, metrics: dict[str, float | int]) -> None:
        """Append one aggregate event and flush its numeric metrics to Parquet."""
        event = {"epoch": epoch, "global_step": step, "group": group, "metrics": metrics}
        for metric, value in metrics.items():
            self.rows.append(
                {
                    "epoch": epoch,
                    "global_step": step,
                    "group": group,
                    "metric": metric,
                    "value": float(value),
                }
            )
        pd.DataFrame(self.rows).to_parquet(self.output_dir / "epoch_metrics.parquet", index=False)
        with (self.output_dir / "events.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(event, allow_nan=False) + "\n")
        with (self.output_dir / "run.log").open("a", encoding="utf-8") as handle:
            handle.write(
                f"epoch={epoch} step={step} group={group} "
                + json.dumps(metrics, allow_nan=False)
                + "\n"
            )


def record_status(
    *, output_dir: Path, status: str, run_id: str, error_type: str | None = None
) -> None:
    """Persist lifecycle events even when training fails before its first metric."""
    event = {
        "status": status,
        "run_id": run_id,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    if error_type is not None:
        event["error_type"] = error_type
    for name in ("events.jsonl", "run.log"):
        with (output_dir / name).open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(event) + "\n")
