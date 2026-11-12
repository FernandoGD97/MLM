"""CPT task names without importing model runtimes."""

from __future__ import annotations

TASKS = {
    "inspect_corpora": ("lab.cpt.inspection.workflow", "inspect_corpora"),
    "evaluate_checkpoint": ("lab.cpt.evaluation.checkpoints", "evaluate_checkpoint"),
    "prepare_corpus": ("lab.cpt.corpus", "prepare_corpus"),
    "pretrain_causal": ("lab.cpt.training.workflow", "pretrain_causal"),
    "pretrain_masked": ("lab.cpt.training.workflow", "pretrain_masked"),
    "compare_pretraining_runs": ("lab.cpt.experiments", "compare_pretraining_runs"),
}
REQUIRES = ("torch", "accelerate", "datasets")
