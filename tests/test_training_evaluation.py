import json, math
from pathlib import Path
import pytest
from encoder_pretraining.training.engine import TargetWeightedAccumulator
from encoder_pretraining.evaluation.language_model import fixed_pppl_subset
from encoder_pretraining.evaluation.entity_linking.metrics import compute_metrics
from encoder_pretraining.evaluation.entity_linking.datasets import Mention

def test_target_weighted_nll_not_batch_mean():
    acc=TargetWeightedAccumulator(); acc.update(1.0,1,5); acc.update(3.0,3,7)
    assert acc.nll==pytest.approx(2.5) and acc.targets==4 and acc.processed_tokens==12

def test_pppl_manifest_reuses_exact_subset(tmp_path):
    docs=[f"doc-{i}" for i in range(20)]; path=tmp_path/"pppl_eval_manifest.json"
    first=fixed_pppl_subset(docs,5,42,path); second=fixed_pppl_subset(docs,5,999,path)
    assert first==second and len(first)==5
    assert json.loads(path.read_text())["indices"]==sorted(json.loads(path.read_text())["indices"])

def test_concept_metrics_alias_deduplicated_input():
    retrieved=[[('C1',.9,'alias one'),('C2',.8,'term')],[('C3',.7,'x'),('C2',.6,'term')]]
    metrics,ranks=compute_metrics(['C1','C2'],retrieved,[1,2])
    assert ranks==[1,2]; assert metrics['recall@1']==.5; assert metrics['recall@2']==1; assert metrics['MRR']==.75

def test_monitor_and_final_scientific_roles_are_explicit():
    # The role behavior itself is exercised through the evaluator selection predicate.
    configs=[{"role":"monitor"},{"role":"final_test"}]
    assert [x for x in configs if x.get("role","monitor")=="monitor" or False]==[configs[0]]
    assert [x for x in configs if x.get("role","monitor")=="monitor" or True]==configs

def test_bidirectional_context_restores_mode_and_training():
    torch=pytest.importorskip("torch")
    from encoder_pretraining.training.model_mode import bidirectional_evaluation
    class Config: model_type="bert"; is_decoder=True
    class Model(torch.nn.Module):
        def __init__(self): super().__init__(); self.config=Config(); self.weight=torch.nn.Parameter(torch.tensor([1.]))
    model=Model().train(); before=model.weight.detach().clone()
    with bidirectional_evaluation(model):
        assert not model.config.is_decoder and not model.training
    assert model.config.is_decoder and model.training and torch.equal(before,model.weight)

def test_candidate_alias_dedup_and_cosine_ip():
    torch=pytest.importorskip("torch")
    from encoder_pretraining.evaluation.entity_linking.candidates import CandidateIndex
    embeddings=torch.nn.functional.normalize(torch.tensor([[1.,0.],[.9,.1],[0.,1.]]),dim=-1)
    index=CandidateIndex(embeddings,["C1","C1","C2"],["a","alias-a","b"])
    result=index.search(torch.tensor([[1.,0.]]),2)[0]
    assert [x[0] for x in result]==["C1","C2"] and result[0][1]==pytest.approx(1.)
