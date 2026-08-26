import random
import pytest
from encoder_pretraining.entities import Span, GazetteerEntityProvider, project_spans
from encoder_pretraining.masking import RandomMasking, EntityFirstMasking, FixedSchedule, LinearSchedule, CosineSchedule, apply_replacement

class Tok:
    mask_token_id=99
    def __len__(self): return 200

def test_gazetteer_and_projection_are_raw_text_based():
    text="Paciente con insuficiencia cardíaca."
    spans=GazetteerEntityProvider(["insuficiencia cardíaca"]).find_entities(text)
    assert spans == [Span(13,35,"insuficiencia cardíaca")]
    assert project_spans(spans,[(0,8),(9,12),(13,25),(26,35),(35,36)]) == [[2,3]]

def test_entity_first_whole_and_budget_control():
    s=EntityFirstMasking(.75).select(range(8),4,random.Random(2),entity_groups=[[1,2],[4,5,6]],tokens=list("abcdefgh"))
    assert len(s.indices)==4 and s.entity_indices=={4,5,6}
    assert not ({1,2}&s.indices)  # no partial second entity
    assert s.whole_entities_selected==1

def test_zero_fraction_equals_random_semantics_and_no_entity_case():
    s=EntityFirstMasking(0).select(range(10),3,random.Random(1),entity_groups=[[2,3]],tokens=[str(i) for i in range(10)])
    assert len(s.indices)==3 and not s.entity_indices and not ({2,3}&s.indices)
    empty=EntityFirstMasking(.5).select(range(5),2,random.Random(1),entity_groups=[],tokens=list("abcde"))
    assert len(empty.indices)==2

def test_random_dynamic_but_seed_reproducible():
    def exposures(seed):
        rng=random.Random(seed); p=RandomMasking()
        return [p.select(range(30),5,rng).indices for _ in range(2)]
    assert exposures(7)==exposures(7)
    assert exposures(7)[0]!=exposures(7)[1]

def test_replacement_policies():
    ids=list(range(20)); selected=set(range(20))
    assert apply_replacement(ids,selected,Tok(),random.Random(0),"100_percent_mask")==[99]*20
    out=apply_replacement(list(range(10000)),set(range(10000)),Tok(),random.Random(4),"bert_80_10_10")
    masked=sum(x==99 for x in out); unchanged=sum(x==i for i,x in enumerate(out))
    assert 7700 < masked < 8300 and 800 < unchanged < 1200

def test_schedules():
    assert FixedSchedule(.15)(.9)==.15
    assert LinearSchedule(.3,.15)(.5)==pytest.approx(.225)
    assert CosineSchedule(.3,.15)(0)==.3 and CosineSchedule(.3,.15)(1)==.15
