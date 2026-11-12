"""Full Accelerate state and explicit epoch-boundary deterministic resumption."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from lab.cpt.training.tracking import write_json


def save_checkpoint(
    *, accelerator: Any, model: Any, tokenizer: Any, path: Path, state: dict[str, Any]
) -> None:
    """Save weights, optimizer, scheduler, scaler and RNG state without overwriting."""
    path.mkdir(parents=True, exist_ok=False)
    model_dir = path / "model"
    accelerator.unwrap_model(model).save_pretrained(model_dir)
    tokenizer.save_pretrained(model_dir)
    accelerator.save_state(str(path / "accelerate"), safe_serialization=True)
    write_json(path=path / "training_state.json", content=state)
