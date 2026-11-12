"""Continual pretraining with explicit scientific policies and artifact contracts."""

from __future__ import annotations

import importlib
from typing import Any

_EXPORTS = {
    "inspect_corpora": "lab.cpt.inspection.workflow",
    "prepare_corpus": "lab.cpt.corpus",
    "tokenize_corpus": "lab.cpt.corpus",
    "score_domain_words": "lab.cpt.importance.domain",
    "project_phrase_scores": "lab.cpt.importance.projection",
    "score_entities": "lab.cpt.importance.entities",
    "extract_keyphrases": "lab.cpt.importance.keyphrases",
    "fuse_scores": "lab.cpt.importance.fusion",
    "select_mask_positions": "lab.cpt.masking.selection",
    "pretrain_causal": "lab.cpt.training.workflow",
    "pretrain_masked": "lab.cpt.training.workflow",
    "evaluate_checkpoint": "lab.cpt.evaluation.checkpoints",
    "evaluate_pretraining": "lab.cpt.evaluation.intrinsic",
    "compare_pretraining_runs": "lab.cpt.experiments",
}
__all__ = sorted(_EXPORTS)


def __getattr__(name: str) -> Any:
    module = _EXPORTS.get(name)
    if module is None:
        raise AttributeError(name)

    return getattr(importlib.import_module(module), name)
