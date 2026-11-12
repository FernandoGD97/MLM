"""Typed single-concern optimization parameters and independent random seeds."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True)
class TrainingParameters:
    """A finite training plan; defaults deliberately describe a small CPU smoke run."""

    epochs: int = 1
    batch_size: int = 2
    gradient_accumulation: int = 1
    learning_rate: float = 5e-5
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    scheduler: Literal["linear", "cosine", "constant"] = "linear"
    warmup_ratio: float = 0.0
    mixed_precision: Literal["no", "fp16", "bf16"] = "no"
    training_seed: int = 42
    data_seed: int = 42
    evaluation_seed: int = 44
    evaluation_probability: float = 0.15
    evaluation_every_epochs: int = 1
    evaluate_pll: bool = False
    tensorboard: bool = False

    def __post_init__(self) -> None:
        integers = (
            self.epochs,
            self.batch_size,
            self.gradient_accumulation,
            self.evaluation_every_epochs,
            self.training_seed,
            self.data_seed,
            self.evaluation_seed,
        )
        if not all(isinstance(value, int) and not isinstance(value, bool) for value in integers):
            raise ValueError("Counts and seeds must be integers, not floats or booleans.")
        if (
            min(
                self.epochs,
                self.batch_size,
                self.gradient_accumulation,
                self.evaluation_every_epochs,
            )
            < 1
        ):
            raise ValueError(
                "Epochs, batch size, accumulation and evaluation frequency must be positive."
            )
        if not all(
            math.isfinite(value)
            for value in (
                self.learning_rate,
                self.weight_decay,
                self.max_grad_norm,
                self.warmup_ratio,
                self.evaluation_probability,
            )
        ):
            raise ValueError("Training numeric parameters must be finite.")
        if self.learning_rate <= 0 or self.weight_decay < 0 or self.max_grad_norm <= 0:
            raise ValueError(
                "Use positive learning_rate/max_grad_norm and nonnegative weight_decay."
            )
        if not 0 <= self.warmup_ratio <= 1 or not 0 < self.evaluation_probability <= 1:
            raise ValueError("Warmup must be in [0,1]; evaluation probability must be in (0,1].")
        if self.mixed_precision not in {"no", "fp16", "bf16"} or self.scheduler not in {
            "linear",
            "cosine",
            "constant",
        }:
            raise ValueError("Unsupported precision or scheduler.")
