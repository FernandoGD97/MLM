"""Same-backbone attention-mode switching and weighted MLM loss."""
def set_attention_mode(model, objective: str):
    """BERT/RoBERTa support causal mode through their native `is_decoder` mask path."""
    from lab.cpt.modeling import set_attention_mode as set_verified_attention_mode
    set_verified_attention_mode(model=model, objective=objective)
    return model

def weighted_mlm_loss(logits, labels, entity_mask=None, entity_loss_weight=1.0):
    import torch
    losses=torch.nn.functional.cross_entropy(logits.view(-1,logits.size(-1)),labels.view(-1),ignore_index=-100,reduction="none").view_as(labels)
    valid=labels.ne(-100); weights=torch.ones_like(losses)
    if entity_mask is not None: weights=torch.where(entity_mask, torch.as_tensor(entity_loss_weight,device=logits.device), weights)
    return (losses*weights)[valid].sum()/weights[valid].sum().clamp_min(1)
