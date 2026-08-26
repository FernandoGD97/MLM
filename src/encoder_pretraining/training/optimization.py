"""Optimizer/scheduler construction shared by CLM and MLM."""

def build_optimizer(model, config):
    import torch
    decay, no_decay = [], []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        (no_decay if name.endswith(("bias", "LayerNorm.weight", "layer_norm.weight")) else decay).append(parameter)
    return torch.optim.AdamW(
        [{"params": decay, "weight_decay": config.get("weight_decay", 0.01)},
         {"params": no_decay, "weight_decay": 0.0}],
        lr=float(config["learning_rate"]), eps=float(config.get("adam_epsilon", 1e-8)),
    )

def build_scheduler(optimizer, config, total_steps):
    from transformers import get_scheduler
    warmup = int(total_steps * float(config.get("warmup_ratio", 0.0)))
    return get_scheduler(config.get("scheduler", "linear"), optimizer, warmup, total_steps)
