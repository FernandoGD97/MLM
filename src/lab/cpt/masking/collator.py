"""Deterministic dynamic masking, explicit corruption and aggregate-only diagnostics."""

from __future__ import annotations

import random
from collections.abc import Mapping, Sequence
from typing import Any

import torch

from lab.cpt.contracts import Window
from lab.cpt.masking.policies import MaskPolicy, WindowSignals
from lab.cpt.masking.selection import select_mask_positions, stable_seed


class MaskCollator:
    """Hold tokenizer/policy/epoch state; never run NER or other detectors inside a batch."""

    def __init__(
        self,
        *,
        tokenizer: Any,
        policy: MaskPolicy,
        signals: Mapping[tuple[str, int], WindowSignals] | None = None,
        objective: str = "mlm",
    ) -> None:
        if tokenizer.pad_token_id is None or tokenizer.mask_token_id is None:
            raise ValueError("Tokenizer requires pad and mask tokens.")
        if objective not in {"mlm", "clm"}:
            raise ValueError("objective must be mlm or clm.")
        self.tokenizer = tokenizer
        self.policy = policy
        self.signals = signals or {}
        self.objective = objective
        self.vocabulary = tuple(
            index for index in range(len(tokenizer)) if index not in tokenizer.all_special_ids
        )
        self.epoch = 0
        self.metrics: list[dict[str, float]] = []
        self.word_types: set[str] = set()
        self.concepts: set[str] = set()

    def __call__(self, windows: Sequence[Window]) -> dict[str, torch.Tensor]:
        maximum = max(len(window.input_ids) for window in windows)
        rows, targets, masks = [], [], []
        vocabulary = self.vocabulary
        for window in windows:
            ids = list(window.input_ids)
            labels = [-100] * len(ids)
            metrics: dict[str, float] = {"eligible_subwords": len(window.eligible)}
            if self.objective == "clm":
                for position in range(len(ids) - 1):
                    if position + 1 in window.eligible:
                        labels[position] = ids[position + 1]
                metrics["selected_targets"] = sum(value != -100 for value in labels)
            else:
                signal = self.signals.get((window.doc_id, window.index))
                selected = select_mask_positions(
                    window=window, policy=self.policy, signals=signal, epoch=self.epoch
                )
                rng = random.Random(
                    stable_seed(
                        seed=self.policy.mask_seed + 1,
                        doc_id=window.doc_id,
                        window_index=window.index,
                        epoch=self.epoch,
                    )
                )
                masked = random_replacements = unchanged = 0
                for unit in selected.units:
                    action = rng.random()
                    for position in unit:
                        labels[position] = ids[position]
                        if action < self.policy.mask_fraction:
                            ids[position] = self.tokenizer.mask_token_id
                            masked += 1
                        elif action < self.policy.mask_fraction + self.policy.random_fraction:
                            if not vocabulary:
                                raise ValueError(
                                    "Random replacement needs nonspecial vocabulary entries."
                                )
                            ids[position] = rng.choice(vocabulary)
                            random_replacements += 1
                        else:
                            unchanged += 1
                metrics.update(
                    {
                        f"disjoint_budget_{index}": value
                        for index, value in enumerate(selected.disjoint_budget)
                    }
                )
                metrics.update(
                    {
                        f"disjoint_targets_{index}": value
                        for index, value in enumerate(selected.disjoint_targets)
                    }
                )
                self.word_types.update(
                    word.text.casefold()
                    for word in window.words
                    if set(word.positions) & set(selected.positions)
                )
                if signal:
                    self.concepts.update(
                        span.concept
                        for span in signal.entities
                        if span.concept is not None
                        and any(
                            span.start <= word.start
                            and word.end <= span.end
                            and set(word.positions) <= set(selected.positions)
                            for word in window.words
                        )
                    )
                metrics.update(
                    selected_targets=len(selected.positions),
                    budget=selected.budget,
                    budget_deviation=selected.deviation,
                    corrupted_positions=sum(
                        before != after for before, after in zip(window.input_ids, ids, strict=True)
                    ),
                    mask_replacements=masked,
                    random_replacements=random_replacements,
                    unchanged_targets=unchanged,
                    selected_words=selected.selected_words,
                    selected_entities=selected.selected_entities,
                    entity_targets=len(selected.entity_positions),
                    selected_units=len(selected.units),
                    total_span_length=sum(map(len, selected.units)),
                    backfill_targets=len(selected.backfill_positions),
                    skipped_spans=selected.skipped_entities,
                    unique_word_types=selected.unique_word_types,
                    unique_concepts=selected.unique_concepts,
                    entity_signal_targets=selected.signal_targets[0],
                    domain_signal_targets=selected.signal_targets[1],
                    keyphrase_signal_targets=selected.signal_targets[2],
                )
            padding = maximum - len(ids)
            rows.append(ids + [self.tokenizer.pad_token_id] * padding)
            targets.append(labels + [-100] * padding)
            masks.append([1] * len(ids) + [0] * padding)
            self.metrics.append(metrics)

        return {
            "input_ids": torch.tensor(rows),
            "labels": torch.tensor(targets),
            "attention_mask": torch.tensor(masks),
        }
