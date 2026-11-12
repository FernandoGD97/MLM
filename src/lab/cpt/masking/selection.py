"""Budget-constrained selection of disjoint subword, word or merged entity units."""

from __future__ import annotations

import hashlib
import math
import random
from dataclasses import dataclass

from lab.cpt.contracts import Window
from lab.cpt.importance.fusion import fuse_scores, percentile_scores
from lab.cpt.masking.policies import MaskPolicy, WindowSignals


@dataclass(frozen=True)
class MaskSelection:
    """Prediction targets, indivisible corruption units and audited budget counts."""

    positions: tuple[int, ...]
    units: tuple[tuple[int, ...], ...]
    budget: int
    deviation: int
    backfill_positions: tuple[int, ...]
    selected_entities: int
    skipped_entities: int
    entity_positions: tuple[int, ...]
    selected_words: int
    unique_word_types: int
    unique_concepts: int
    mean_span_length: float
    signal_targets: tuple[int, int, int]
    disjoint_budget: tuple[int, int, int] = (0, 0, 0)
    disjoint_targets: tuple[int, int, int] = (0, 0, 0)


def stable_seed(*, seed: int, doc_id: str, window_index: int, epoch: int) -> int:
    """Derive order/worker-independent random state using SHA-256, never Python hash."""
    key = f"{seed}:{len(doc_id)}:{doc_id}:{window_index}:{epoch}".encode()

    return int.from_bytes(hashlib.sha256(key).digest()[:8], "big")


def select_mask_positions(
    *, window: Window, policy: MaskPolicy, signals: WindowSignals | None = None, epoch: int = 0
) -> MaskSelection:
    """Reach floor(p*N) exactly when compatible units exist; never split an entity.

    Overlapping whole entities merge into indivisible connected components. Exact subset
    feasibility guides ranked selection. Underfill is opt-in and records the deviation.
    Non-entity backfill in M2 is explicitly at subword granularity.
    """
    signals = signals or WindowSignals()
    random_state = random.Random(
        stable_seed(
            seed=policy.mask_seed, doc_id=window.doc_id, window_index=window.index, epoch=epoch
        )
    )
    budget = math.floor(policy.probability * len(window.eligible))
    scores, detector_scores = _word_scores(window, policy, signals)
    entities = []
    for span in signals.entities:
        aligned_words = [
            word for word in window.words if span.start <= word.start and word.end <= span.end
        ]
        if (
            not aligned_words
            or aligned_words[0].start != span.start
            or aligned_words[-1].end != span.end
        ):
            raise ValueError(
                "Entity spans must cover complete aligned words; "
                "discard/count truncation explicitly."
            )
        positions = frozenset(
            position
            for word in window.words
            if span.start <= word.start and word.end <= span.end
            for position in word.positions
        )
        if positions:
            entities.append((span, positions))
    entity_positions = set().union(*(positions for _, positions in entities)) if entities else set()
    if policy.policy == "M0" and policy.random_unit == "subword":
        units = [(position,) for position in window.eligible]
    elif policy.policy == "M2":
        groups = [set(positions) for _, positions in entities]
        merged: list[set[int]] = []
        for group in groups:
            crossing = [other for other in merged if other & group]
            for other in crossing:
                group |= other
                merged.remove(other)
            merged.append(group)
        units = [tuple(sorted(group)) for group in merged]
        units.extend(
            (position,) for position in window.eligible if position not in entity_positions
        )
    else:
        units = [word.positions for word in window.words]
    by_position = {
        position: index for index, word in enumerate(window.words) for position in word.positions
    }
    random_state.shuffle(units)
    ranked = []
    for unit in units:
        word_ids = {by_position[position] for position in unit}
        score = sum(scores[index] for index in word_ids) / len(word_ids)
        if policy.policy == "M2":
            score = max(
                (span.score for span, positions in entities if positions & set(unit)), default=0.0
            )
        ranked.append((unit, score))
    ranked.sort(key=lambda item: -item[1])
    disjoint_budget = disjoint_targets = (0, 0, 0)
    if policy.fusion == "disjoint" and policy.policy in {"M5", "M6", "M7"}:
        selected, disjoint_budget, disjoint_targets = _allocate_disjoint(
            ranked, detector_scores, by_position, policy, budget
        )
    else:
        selected = select_budget_units([unit for unit, _ in ranked], budget, policy.budget_mode)
    positions = set(position for unit in selected for position in unit)
    backfill = {
        position for unit, score in ranked if score <= 0 and unit in selected for position in unit
    }
    if policy.policy == "M0":
        backfill = set()
    selected_words = [word for word in window.words if positions.intersection(word.positions)]
    selected_entities = [(span, group) for span, group in entities if group <= positions]
    concepts = {span.concept for span, _ in selected_entities if span.concept is not None}
    contributions = tuple(
        sum(
            len(set(word.positions) & positions)
            for index, word in enumerate(window.words)
            if detector[index] > 0
        )
        for detector in detector_scores
    )

    return MaskSelection(
        tuple(sorted(positions)),
        tuple(selected),
        budget,
        len(positions) - budget,
        tuple(sorted(backfill)),
        len(selected_entities),
        len(entities) - len(selected_entities) + signals.skipped_spans,
        tuple(sorted(entity_positions & positions)),
        len(selected_words),
        len({word.text.casefold() for word in selected_words}),
        len(concepts),
        sum(len(unit) for unit in selected) / max(1, len(selected)),
        contributions,
        disjoint_budget,
        disjoint_targets,
    )


def select_budget_units(
    units: list[tuple[int, ...]], budget: int, mode: str
) -> list[tuple[int, ...]]:
    """Select ranked disjoint units under an exact or explicit underfill target budget."""
    if budget < 0 or mode not in {"strict", "underfill"}:
        raise ValueError("Use a nonnegative budget and mode=strict/underfill.")
    flattened = [position for unit in units for position in unit]
    if any(not unit for unit in units) or len(flattened) != len(set(flattened)):
        raise ValueError("Budget units must be nonempty and disjoint.")
    suffix = [1] * (len(units) + 1)
    limit = (1 << (budget + 1)) - 1
    for index in range(len(units) - 1, -1, -1):
        suffix[index] = (suffix[index + 1] | suffix[index + 1] << len(units[index])) & limit
    target = budget
    if not (suffix[0] >> target) & 1:
        if mode == "strict":
            raise ValueError(
                f"Exact budget B={budget} is impossible with unit sizes "
                f"{[len(unit) for unit in units][:20]}; choose "
                "budget_mode='underfill' or another probability."
            )
        target = suffix[0].bit_length() - 1
    selected = []
    for index, unit in enumerate(units):
        remaining = target - len(unit)
        if remaining >= 0 and (suffix[index + 1] >> remaining) & 1:
            selected.append(unit)
            target = remaining

    return selected


def _word_scores(
    window: Window, policy: MaskPolicy, signals: WindowSignals
) -> tuple[tuple[float, ...], tuple[tuple[float, ...], ...]]:
    count = len(window.words)
    detectors = tuple(
        tuple(value) if value else (0.0,) * count
        for value in (signals.entity, signals.domain, signals.keyphrase)
    )
    if any(
        len(value) != count or not all(math.isfinite(score) for score in value)
        for value in detectors
    ):
        raise ValueError("Importance signals must align one-to-one with word occurrences.")
    active = {
        "M0": (),
        "M1": (0,),
        "M2": (0,),
        "M3": (1,),
        "M4": (2,),
        "M5": (0, 1),
        "M6": (0, 2),
        "M7": (0, 1, 2),
        "M8": (),
    }[policy.policy]
    if policy.policy == "M8":
        if len(signals.uncertainty) != count or not all(
            math.isfinite(value) for value in signals.uncertainty
        ):
            raise ValueError("M8 requires finite, checkpoint-specific cached word uncertainty.")
        return signals.uncertainty, detectors
    if not active:
        return (0.0,) * count, detectors
    if policy.fusion in {"hierarchical", "disjoint"}:
        normalized = [percentile_scores(scores=detectors[index]) for index in active]
        if policy.fusion == "hierarchical":
            scores = tuple(
                sum(
                    (2 ** (len(active) - rank)) + signal[index]
                    for rank, signal in enumerate(normalized)
                    if detectors[active[rank]][index] > 0
                )
                for index in range(count)
            )
        else:
            owners = [
                next(
                    (
                        rank
                        for rank, detector in enumerate(active)
                        if detectors[detector][index] > 0
                    ),
                    None,
                )
                for index in range(count)
            ]
            scores = _disjoint_priorities(
                window, normalized, owners, tuple(policy.weights[index] for index in active)
            )
    else:
        scores = fuse_scores(
            signals=[detectors[index] for index in active],
            weights=[policy.weights[index] for index in active],
            method=policy.fusion,
        )

    return scores, detectors


def _disjoint_priorities(
    window: Window,
    signals: list[tuple[float, ...]],
    owners: list[int | None],
    weights: tuple[float, ...],
) -> tuple[float, ...]:
    pools = [
        [index for index, owner in enumerate(owners) if owner == detector]
        for detector in range(len(signals))
    ]
    for detector, pool in enumerate(pools):
        pool.sort(key=lambda index: (-signals[detector][index], index))
    allocated = [0] * len(signals)
    scores = [0.0] * len(owners)
    rank = len(owners)
    while any(pool and weight > 0 for pool, weight in zip(pools, weights, strict=True)):
        detector = min(
            (index for index, pool in enumerate(pools) if pool and weights[index] > 0),
            key=lambda index: allocated[index] / weights[index],
        )
        word = pools[detector].pop(0)
        scores[word] = float(rank)
        rank -= 1
        allocated[detector] += len(window.words[word].positions)

    return tuple(scores)


def _allocate_disjoint(
    ranked: list[tuple[tuple[int, ...], float]],
    detectors: tuple[tuple[float, ...], ...],
    by_position: dict[int, int],
    policy: MaskPolicy,
    budget: int,
) -> tuple[list[tuple[int, ...]], tuple[int, ...], tuple[int, ...]]:
    active = {"M5": (0, 1), "M6": (0, 2), "M7": (0, 1, 2)}[policy.policy]
    total_weight = sum(policy.weights[index] for index in active)
    if total_weight <= 0:
        raise ValueError("Active disjoint detectors need a positive total weight.")
    shares = [
        budget * policy.weights[index] / total_weight if index in active else 0
        for index in range(3)
    ]
    quotas = [math.floor(share) for share in shares]
    for index in sorted(active, key=lambda index: (-(shares[index] - quotas[index]), index))[
        : budget - sum(quotas)
    ]:
        quotas[index] += 1
    pools = [[], [], []]
    for unit, _ in ranked:
        owner = next(
            (
                index
                for index in active
                if policy.weights[index] > 0
                and any(detectors[index][by_position[position]] > 0 for position in unit)
            ),
            None,
        )
        if owner is not None:
            pools[owner].append(unit)
    selected = []
    allocations = [0, 0, 0]
    for index in active:
        chosen = select_budget_units(pools[index], quotas[index], "underfill")
        allocations[index] = sum(map(len, chosen))
        selected.extend(chosen)
    remainder = budget - sum(map(len, selected))
    unused = [unit for unit, _ in ranked if unit not in selected]
    try:
        backfill = select_budget_units(unused, remainder, policy.budget_mode)
    except ValueError:
        selected = select_budget_units([unit for unit, _ in ranked], budget, policy.budget_mode)
        allocations = [
            sum(len(unit) for unit in selected if unit in pools[index]) for index in range(3)
        ]
        return selected, tuple(quotas), tuple(allocations)
    selected.extend(backfill)
    allocations = [
        sum(len(unit) for unit in selected if unit in pools[index]) for index in range(3)
    ]
    return selected, tuple(quotas), tuple(allocations)
