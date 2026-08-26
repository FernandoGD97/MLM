"""Dynamic MLM and CLM batch collation."""
import random
from .masking import RandomMasking, apply_replacement
from .entities import project_spans

class MLMCollator:
    def __init__(self, tokenizer, schedule, policy=None, entity_provider=None, replacement="bert_80_10_10", seed=0, include_metadata=False):
        self.tokenizer, self.schedule = tokenizer, schedule
        self.policy, self.entity_provider, self.replacement = policy or RandomMasking(), entity_provider, replacement
        self.seed=seed; self.rng=random.Random(seed); self.progress=0.0; self.include_metadata=include_metadata; self.last_diagnostics={}
    def reset_seed(self): self.rng.seed(self.seed)
    def set_progress(self, processed_tokens, token_budget): self.progress=min(1,processed_tokens/token_budget) if token_budget else 0
    def __call__(self, examples):
        import torch
        rows=[]
        for ex in examples:
            ids=list(ex["input_ids"]); special=ex.get("special_tokens_mask") or self.tokenizer.get_special_tokens_mask(ids, already_has_special_tokens=True)
            eligible=[i for i,x in enumerate(special) if not x]
            budget=min(len(eligible), round(self.schedule(self.progress)*len(eligible)))
            spans=self.entity_provider.find_entities(ex["text"]) if self.entity_provider else []
            groups=project_spans(spans, ex.get("offset_mapping", []))
            tokens=self.tokenizer.convert_ids_to_tokens(ids)
            selection=self.policy.select(eligible,budget,self.rng,entity_groups=groups,tokens=tokens)
            labels=[x if i in selection.indices else -100 for i,x in enumerate(ids)]
            rows.append((apply_replacement(ids,selection.indices,self.tokenizer,self.rng,self.replacement),labels,selection,len(eligible)))
        maxlen=max(len(x[0]) for x in rows); pad=self.tokenizer.pad_token_id
        def padded(x,v): return x+[v]*(maxlen-len(x))
        selections=[x[2] for x in rows]; eligible=sum(x[3] for x in rows); masked=sum(len(x.indices) for x in selections)
        diagnostics={"processed_tokens":eligible,"eligible_tokens":eligible,"masked_tokens":masked,
          "effective_mask_ratio":masked/eligible if eligible else 0.0,
          "masked_entity_tokens":sum(len(x.entity_indices) for x in selections),
          "masked_generic_tokens":sum(len(x.indices-x.entity_indices) for x in selections),
          "whole_entities_selected":sum(x.whole_entities_selected for x in selections),
          "entity_mask_fraction":sum(len(x.entity_indices) for x in selections)/masked if masked else 0.0,
          "special_tokens_masked":0,"CLM_prediction_targets":0,"MLM_prediction_targets":masked}
        self.last_diagnostics=diagnostics
        batch={"input_ids":torch.tensor([padded(x[0],pad) for x in rows]), "labels":torch.tensor([padded(x[1],-100) for x in rows]),
               "attention_mask":torch.tensor([padded([1]*len(x[0]),0) for x in rows])}
        if self.include_metadata: batch.update(masking_metadata=selections,diagnostics=diagnostics)
        return batch

class CLMCollator:
    """MLM heads do not shift internally: position i targets token i+1."""
    def __init__(self, tokenizer, include_metadata=False): self.tokenizer=tokenizer; self.include_metadata=include_metadata; self.last_diagnostics={}
    def __call__(self, examples):
        import torch
        n=max(map(lambda x:len(x["input_ids"]),examples)); pad=self.tokenizer.pad_token_id
        ids=[list(x["input_ids"])+[pad]*(n-len(x["input_ids"])) for x in examples]
        masks=[[1]*len(x["input_ids"])+[0]*(n-len(x["input_ids"])) for x in examples]
        labels=[]
        for row,m in zip(ids,masks):
            target=[row[i+1] if i+1 < n and m[i+1] else -100 for i in range(n)]
            labels.append(target)
        targets=sum(v!=-100 for row in labels for v in row)
        self.last_diagnostics={"processed_tokens":sum(map(sum,masks)),"CLM_prediction_targets":targets,"MLM_prediction_targets":0}
        batch={"input_ids":torch.tensor(ids),"attention_mask":torch.tensor(masks),"labels":torch.tensor(labels)}
        if self.include_metadata: batch["diagnostics"]=self.last_diagnostics
        return batch
