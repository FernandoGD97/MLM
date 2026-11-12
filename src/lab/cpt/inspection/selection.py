"""Corpus-wide word budgets with proportional source quotas and indivisible spans."""

from __future__ import annotations

import math
import random
from collections.abc import Sequence
from dataclasses import dataclass

from lab.cpt.importance.fusion import fuse_scores
from lab.cpt.inspection.corpus import Document
from lab.cpt.masking.selection import select_budget_units

TECHNIQUES = (
    "random",
    "terms",
    "entities",
    "whole_entities",
    "domain",
    "keyphrase",
    "ner_domain",
    "ner_keyphrase",
    "triple",
    "entropy",
    "attention",
)
ALIASES = dict(
    zip(
        (f"M{i}" for i in range(9)),
        (
            "random",
            "entities",
            "whole_entities",
            "domain",
            "keyphrase",
            "ner_domain",
            "ner_keyphrase",
            "triple",
            "entropy",
        ),
        strict=True,
    )
)


@dataclass(frozen=True)
class Selection:
    """Global selected word IDs and requested/realized per-corpus allocations."""

    positions: tuple[int, ...]
    requested: int
    quotas: tuple[float, ...]
    allocated: tuple[int, ...]
    backfill_words: int


def select_words(
    *,
    documents: Sequence[Document],
    signals: Sequence[Sequence[Sequence[float]]],
    technique: str,
    probability: float,
    random_state: int = 42,
    backfill: bool = True,
    budget_mode: str = "strict",
    entropy: Sequence[Sequence[float]] | None = None,
    attention: Sequence[Sequence[float]] | None = None,
) -> Selection:
    """Select floor(p * all eligible words), minimizing squared source-quota deviations."""
    if technique not in TECHNIQUES or not math.isfinite(probability) or not 0 <= probability <= 1:
        raise ValueError("Use a known technique and probability within [0, 1].")
    sizes = [
        sum(len(doc.words) for doc in documents if doc.corpus == corpus)
        for corpus in range(max(doc.corpus for doc in documents) + 1)
    ]
    total = sum(sizes)
    budget = math.floor(probability * total)
    quotas = tuple(budget * size / total if total else 0.0 for size in sizes)
    pools: list[list[tuple[int, ...]]] = [[] for _ in sizes]
    scores_by_position = {}
    offset = 0
    rng = random.Random(random_state)
    for document_index, (doc, evidence) in enumerate(zip(documents, signals, strict=True)):
        if technique == "random":
            scores = (0.0,) * len(doc.words)
        elif technique == "attention":
            if attention is None:
                raise ValueError("attention requires an explicit local encoder checkpoint.")
            scores = attention[document_index]
        elif technique == "entropy":
            if entropy is None:
                raise ValueError("entropy requires an explicit local base_model.")
            scores = entropy[document_index]
        elif technique in {"ner_domain", "ner_keyphrase", "triple"}:
            active = {"ner_domain": (1, 2), "ner_keyphrase": (1, 3), "triple": (1, 2, 3)}[technique]
            scores = fuse_scores(
                signals=[evidence[index] for index in active], weights=[1.0] * len(active)
            )
        else:
            index = {"terms": 0, "entities": 1, "whole_entities": 1, "domain": 2, "keyphrase": 3}[
                technique
            ]
            scores = evidence[index]
        units = [{index} for index in range(len(doc.words))]
        if technique == "whole_entities":
            for start, end in doc.entities:
                group = {
                    index
                    for index, (left, right) in enumerate(doc.words)
                    if left < end and right > start
                }
                if not group:
                    continue
                crossing = [unit for unit in units if unit & group]
                for unit in crossing:
                    group |= unit
                    units.remove(unit)
                units.append(group)
        for index, score in enumerate(scores):
            scores_by_position[offset + index] = score
        for unit in units:
            if backfill or technique == "random" or any(scores[index] > 0 for index in unit):
                pools[doc.corpus].append(tuple(offset + index for index in sorted(unit)))
        offset += len(doc.words)
    for pool in pools:
        rng.shuffle(pool)
        pool.sort(key=lambda unit: -sum(scores_by_position[index] for index in unit) / len(unit))
    allocations = proportional_budgets(pools=pools, quotas=quotas, budget=budget, mode=budget_mode)
    selected = [
        position
        for pool, count in zip(pools, allocations, strict=True)
        for unit in select_budget_units(pool, count, "strict")
        for position in unit
    ]

    return Selection(
        tuple(sorted(selected)),
        budget,
        quotas,
        allocations,
        sum(scores_by_position[index] <= 0 for index in selected) if technique != "random" else 0,
    )


def proportional_budgets(
    *,
    pools: Sequence[Sequence[tuple[int, ...]]],
    quotas: Sequence[float],
    budget: int,
    mode: str,
) -> tuple[int, ...]:
    """Find the closest feasible proportional allocation at the requested global budget.

    Ordinary word masking uses largest remainder directly. Constrained whole-span cases
    use a finite exact dynamic program; underfill chooses the largest feasible total first.
    """
    if mode not in {"strict", "underfill"}:
        raise ValueError("budget_mode must be strict or underfill.")
    reachable = []
    for pool in pools:
        bits = 1
        for unit in pool:
            bits |= bits << len(unit)
            bits &= (1 << (budget + 1)) - 1
        reachable.append(bits)
    counts = [math.floor(quota) for quota in quotas]
    for index in sorted(range(len(quotas)), key=lambda i: (-(quotas[i] - counts[i]), i))[
        : budget - sum(counts)
    ]:
        counts[index] += 1
    if all((bits >> count) & 1 for bits, count in zip(reachable, counts, strict=True)):
        return tuple(counts)
    states: dict[int, tuple[float, tuple[int, ...]]] = {0: (0.0, ())}
    for bits, quota in zip(reachable, quotas, strict=True):
        choices = [value for value in range(budget + 1) if (bits >> value) & 1]
        updated = {}
        for subtotal, (cost, allocation) in states.items():
            for value in choices:
                if subtotal + value > budget:
                    break
                candidate = (cost + (value - quota) ** 2, (*allocation, value))
                if subtotal + value not in updated or candidate < updated[subtotal + value]:
                    updated[subtotal + value] = candidate
        states = updated
    if budget not in states and mode == "strict":
        raise ValueError(
            "Requested global budget is impossible; use backfill=True or "
            "budget_mode='underfill', or change probability."
        )

    return states[budget if budget in states else max(states)][1]
