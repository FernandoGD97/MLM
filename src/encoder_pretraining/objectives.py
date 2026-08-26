"""Same-backbone attention-mode switching and weighted MLM loss."""
def set_attention_mode(model, objective: str):
    """BERT/RoBERTa support causal mode through their native `is_decoder` mask path."""
    if objective not in {"clm","mlm"}: raise ValueError(objective)
    model_type=getattr(model.config,"model_type",None)
    if model_type not in {"bert","roberta"}:
        raise ValueError(f"{model_type!r} is not verified for safe same-backbone causal attention; refusing to continue")
    causal=objective=="clm"
    model.config.is_decoder=causal
    for module in model.modules():
        if hasattr(module,"config") and hasattr(module.config,"is_decoder"): module.config.is_decoder=causal
    return model

def weighted_mlm_loss(logits, labels, entity_mask=None, entity_loss_weight=1.0):
    import torch
    losses=torch.nn.functional.cross_entropy(logits.view(-1,logits.size(-1)),labels.view(-1),ignore_index=-100,reduction="none").view_as(labels)
    valid=labels.ne(-100); weights=torch.ones_like(losses)
    if entity_mask is not None: weights=torch.where(entity_mask, torch.as_tensor(entity_loss_weight,device=logits.device), weights)
    return (losses*weights)[valid].sum()/weights[valid].sum().clamp_min(1)
