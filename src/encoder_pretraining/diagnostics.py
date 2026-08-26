from dataclasses import dataclass, asdict
@dataclass
class MaskingStats:
    processed_tokens:int=0; eligible_tokens:int=0; masked_tokens:int=0
    masked_entity_tokens:int=0; masked_generic_tokens:int=0; whole_entities_selected:int=0
    special_tokens_masked:int=0; CLM_prediction_targets:int=0; MLM_prediction_targets:int=0
    @property
    def effective_mask_ratio(self): return self.masked_tokens/self.eligible_tokens if self.eligible_tokens else 0
    @property
    def entity_mask_fraction(self): return self.masked_entity_tokens/self.masked_tokens if self.masked_tokens else 0
    def to_dict(self): return {**asdict(self),"effective_mask_ratio":self.effective_mask_ratio,"entity_mask_fraction":self.entity_mask_fraction}
