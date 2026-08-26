"""WHAT-to-mask policies, HOW-MUCH schedules, and HOW-to-replace policies."""
from dataclasses import dataclass
import math, random, string

class FixedSchedule:
    def __init__(self, probability): self.probability = probability
    def __call__(self, progress): return self.probability
class LinearSchedule:
    def __init__(self, start, end): self.start, self.end = start, end
    def __call__(self, progress): return self.start + (self.end-self.start)*min(1,max(0,progress))
class CosineSchedule(LinearSchedule):
    def __call__(self, progress):
        p=min(1,max(0,progress)); return self.end+(self.start-self.end)*(1+math.cos(math.pi*p))/2

@dataclass
class Selection:
    indices: set[int]; entity_indices: set[int]; origins: dict[int, str]; whole_entities_selected: int = 0

class RandomMasking:
    def select(self, eligible, budget, rng, **kwargs):
        chosen=set(rng.sample(list(eligible), min(budget,len(eligible))))
        return Selection(chosen,set(),{i:"RANDOM" for i in chosen})

class EntityFirstMasking:
    """EXPERIMENTAL DESIGN CHOICE: a strict entity sub-budget preserves total compute."""
    def __init__(self, fraction=.5, remainder="random", stopwords=()):
        if not 0 <= fraction <= 1: raise ValueError("entity budget fraction must be in [0,1]")
        self.fraction, self.remainder, self.stopwords = fraction, remainder, {x.casefold() for x in stopwords}
    def select(self, eligible, budget, rng, entity_groups=(), tokens=()):
        entity_cap=round(budget*self.fraction); selected=set(); count=0
        groups=[set(g)&set(eligible) for g in entity_groups]; rng.shuffle(groups)
        all_entity=set().union(*groups) if groups else set()
        for group in groups:
            if group and len(group) <= entity_cap-len(selected) and not group&selected:
                selected |= group; count += 1
        origins={i:"ENTITY" for i in selected}; remaining=budget-len(selected)
        generic=list(set(eligible)-all_entity-selected)
        preferred=generic
        if self.remainder == "content_random":
            preferred=[i for i in generic if tokens[i].casefold() not in self.stopwords and not all(c in string.punctuation for c in tokens[i])]
        rng.shuffle(preferred); add=preferred[:remaining]
        if len(add)<remaining:
            fallback=list(set(generic)-set(add)); rng.shuffle(fallback); add += fallback[:remaining-len(add)]
        selected.update(add); origins.update({i:("CONTENT" if self.remainder=="content_random" else "RANDOM") for i in add})
        return Selection(selected, selected & all_entity, origins, count)

def apply_replacement(ids, selected, tokenizer, rng, policy):
    output=list(ids)
    for i in selected:
        if policy == "100_percent_mask": output[i]=tokenizer.mask_token_id
        elif policy == "bert_80_10_10":
            x=rng.random()
            if x < .8: output[i]=tokenizer.mask_token_id
            elif x < .9: output[i]=rng.randrange(len(tokenizer))
        else: raise ValueError(f"Unknown replacement policy: {policy}")
    return output
