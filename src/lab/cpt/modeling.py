"""Verified BERT/RoBERTa causal conversion and content-addressed checkpoints."""

from __future__ import annotations

import hashlib
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator, Literal

import torch


def set_attention_mode(*, model: Any, objective: Literal["clm", "mlm"]) -> None:
    """Switch both configuration and cached attention-module state on supported encoders."""
    if objective not in {"clm", "mlm"}:
        raise ValueError("objective must be clm or mlm.")
    if model.config.model_type not in {"bert", "roberta"}:
        raise ValueError(
            f"Unsupported causal encoder {model.config.model_type!r}; "
            "only BERT/RoBERTa are verified."
        )
    if getattr(model.config, "add_cross_attention", False):
        raise ValueError("Cross-attention models are outside the verified encoder path.")
    causal = objective == "clm"
    if hasattr(model, "set_attn_implementation"):
        model.set_attn_implementation("eager")
    for module in model.modules():
        if hasattr(module, "config"):
            module.config.is_decoder = causal
            module.config.use_cache = False
            module.config._attn_implementation = "eager"
        if hasattr(module, "is_decoder"):
            module.is_decoder = causal
        if hasattr(module, "is_causal"):
            module.is_causal = causal
        if hasattr(module, "attn_implementation"):
            module.attn_implementation = "eager"


def verify_attention(*, model: Any, tokenizer: Any, objective: str) -> None:
    """Perturb future tokens and verify the intended context dependency before training."""
    if len(tokenizer) != model.get_input_embeddings().num_embeddings:
        raise ValueError("Tokenizer vocabulary and model embedding count disagree.")
    vocabulary = [
        index for index in range(len(tokenizer)) if index not in tokenizer.all_special_ids
    ]
    if len(vocabulary) < 3:
        raise ValueError("Attention verification needs at least three nonspecial token IDs.")
    device = next(model.parameters()).device
    first = torch.tensor(
        [[vocabulary[0], vocabulary[1], vocabulary[0], vocabulary[1]]], device=device
    )
    second = first.clone()
    second[:, 2:] = vocabulary[2]
    was_training = model.training
    model.eval()
    try:
        with torch.no_grad():
            left = model(input_ids=first, attention_mask=torch.ones_like(first)).logits[:, :2]
            right = model(input_ids=second, attention_mask=torch.ones_like(second)).logits[:, :2]
        same = torch.allclose(left, right, atol=1e-6, rtol=1e-5)
        if (objective == "clm" and not same) or (objective == "mlm" and same):
            raise ValueError(
                f"{objective} attention verification failed; refuse to train this checkpoint."
            )
    finally:
        model.train(was_training)


@contextmanager
def bidirectional_attention(*, model: Any) -> Iterator[None]:
    """Restore attention and training mode even when evaluation raises."""
    objective = "clm" if model.config.is_decoder else "mlm"
    was_training = model.training
    set_attention_mode(model=model, objective="mlm")
    try:
        yield
    finally:
        set_attention_mode(model=model, objective=objective)
        model.train(was_training)


def checkpoint_sha256(*, model_dir: str | Path) -> str:
    """Hash all model/tokenizer files by relative path and bytes, excluding training state."""
    root = Path(model_dir)
    if not root.is_dir() or not (root / "config.json").is_file():
        raise ValueError(f"{root} is not a saved model/tokenizer directory.")
    digest = hashlib.sha256()
    for path in sorted(root.rglob("*")):
        if path.is_file():
            digest.update(path.relative_to(root).as_posix().encode())
            with path.open("rb") as handle:
                for block in iter(lambda: handle.read(1 << 20), b""):
                    digest.update(block)

    return digest.hexdigest()
