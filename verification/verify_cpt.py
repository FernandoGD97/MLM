"""Offline synthetic invariants for the CPT Python/YAML surface and complete training path."""

from __future__ import annotations

import dataclasses
import json
import math
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import pandas as pd
import torch
import yaml
from tokenizers import Tokenizer, models, pre_tokenizers, processors
from transformers import BertConfig, BertForMaskedLM, PreTrainedTokenizerFast

from _harness import Checks, run
from lab.core.tasks import list_tasks, resolve_task
from lab.cpt.contracts import ImportanceResult, Span, Window, Word
from lab.cpt.corpus import grouped_split, prepare_corpus, tokenize_corpus
from lab.cpt.evaluation.intrinsic import evaluate_pretraining, pseudo_likelihood
from lab.cpt.experiments import compare_pretraining_runs
from lab.cpt.importance.domain import score_domain_words
from lab.cpt.importance.entities import score_entities
from lab.cpt.importance.fusion import fuse_scores, percentile_scores
from lab.cpt.importance.keyphrases import extract_keyphrases
from lab.cpt.importance.projection import project_phrase_scores
from lab.cpt.importance.uncertainty import predictive_entropy
from lab.cpt.masking.collator import MaskCollator
from lab.cpt.masking.policies import MaskPolicy, WindowSignals
from lab.cpt.masking.selection import select_mask_positions
from lab.cpt.modeling import checkpoint_sha256, set_attention_mode, verify_attention
from lab.cpt.training.arguments import TrainingParameters
from lab.cpt.training.workflow import pretrain_causal, pretrain_masked

TEXT = "severe chronic heart failure without pulmonary edema"


def fixture(root: Path):
    vocabulary = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]", "severe", "chronic", "heart", "failure", "without", "pul", "##monary", "edema", "no", "today", "pain", "patient", "stable"]
    backend = Tokenizer(models.WordPiece(vocab={word: index for index, word in enumerate(vocabulary)}, unk_token="[UNK]"))
    backend.pre_tokenizer = pre_tokenizers.BertPreTokenizer()
    backend.post_processor = processors.TemplateProcessing(single="[CLS] $A [SEP]", special_tokens=[("[CLS]", 2), ("[SEP]", 3)])
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=backend, unk_token="[UNK]", pad_token="[PAD]", cls_token="[CLS]", sep_token="[SEP]", mask_token="[MASK]")
    base = root / "base"
    torch.manual_seed(11)
    model = BertForMaskedLM(BertConfig(vocab_size=len(tokenizer), hidden_size=16, num_hidden_layers=1, num_attention_heads=2, intermediate_size=32, max_position_embeddings=64, hidden_dropout_prob=0.1, attention_probs_dropout_prob=0.1))
    model.save_pretrained(base)
    tokenizer.save_pretrained(base)
    training = pd.DataFrame([{"doc_id": f"train-{index}", "text": TEXT + (" today" if index % 2 else " no"), "source": "synthetic", "translation_status": "native"} for index in range(4)])
    validation = pd.DataFrame([{"doc_id": "dev", "text": TEXT + " patient stable"}])
    entity_rows = [{"filename": row.doc_id, "label": "DISEASE", "start_span": 15, "end_span": 28, "text": "heart failure", "score": 0.8, "code": "C1"} for row in training.itertuples()]

    return tokenizer, model, base, training, validation, pd.DataFrame(entity_rows)


def contracts(checks: Checks, tokenizer, training, predictions):
    prepared = prepare_corpus(documents=training.assign(entities_json='[{"gold":true}]', n_entities=3))
    checks.equal("gold annotations removed", prepared.entities_json.tolist(), ["[]"] * 4)
    checks.check("metadata retained", prepared.source.eq("synthetic").all())
    checks.raises("integer IDs rejected", ValueError, prepare_corpus, documents=training.assign(doc_id=range(4)))
    windows = tokenize_corpus(documents=training, tokenizer=tokenizer, max_length=32)
    window = windows[0]
    from encoder_pretraining.data import Document,tokenize_and_chunk
    legacy,_=tokenize_and_chunk([Document("legacy",TEXT,{})],tokenizer,max_length=32)
    checks.check("legacy fast tokenization regression",len(legacy)==1 and legacy[0]["input_ids"][0]==tokenizer.cls_token_id)
    pulmonary = next(word for word in window.words if word.text == "pulmonary")
    checks.equal("subword fragmentation kept", len(pulmonary.positions), 2)
    checks.check("special tokens excluded", all(window.input_ids[position] not in tokenizer.all_special_ids for position in window.eligible))
    spans = ImportanceResult((2.0, 4.0), (Span(15, 28, 2), Span(21, 28, 4)), "span_occurrence", "synthetic")
    for method, expected in [("maximum", (2, 4)), ("mean", (2, 3)), ("sum", (2, 6)), ("uniform", (1, 5)), ("length_normalized", (10/12, 14/12 + 4))]:
        projected = project_phrase_scores(words=window.words, importance=spans, method=method)
        actual = projected.scores[2:4]
        checks.check(f"phrase projection {method}", all(abs(a-b) < 1e-8 for a,b in zip(actual, expected)))
    checks.raises("nonfinite scores rejected", ValueError, Span, 0, 2, float("nan"))
    checks.raises("misaligned phrase rejected", ValueError, project_phrase_scores, words=window.words, importance=ImportanceResult((1.0,), (Span(16,28),), "span_occurrence", "bad"), method="sum")
    entity = score_entities(spans=predictions.iloc[:1], text=training.text.iloc[0], mode="confidence", type_weights={"DISEASE": 2})
    checks.equal("confidence and entity type", entity.scores, (1.6,))
    checks.equal("binary entity scoring", score_entities(spans=predictions.iloc[:1], text=training.text.iloc[0]).scores, (1.0,))
    for method in ("tfidf", "tficf", "log_odds", "frequency_ratio"):
        result = score_domain_words(training_texts=[TEXT, TEXT], reference_texts=["patient stable no pain"], method=method)
        checks.check(f"finite {method}", bool(result.scores) and all(math.isfinite(value) for _,value in result.scores))
    log_odds = dict(score_domain_words(training_texts=["heart heart heart no", "heart"], reference_texts=["no no stable"], method="log_odds", minimum_frequency=1).scores)
    checks.check("domain odds direction", log_odds["heart"] > log_odds["no"])
    checks.check("keyphrase extraction has actual aligned candidates", bool(extract_keyphrases(text=TEXT, word_scores={"heart": 2, "failure": 3}).spans))
    checks.equal("percentile ties", percentile_scores(scores=[0,1,1]), (0,2/3,2/3))
    for method in ("score", "weighted_rank", "rrf"):
        fused = fuse_scores(signals=[[0,2,1],[0,1,2]],weights=[1,1],method=method)
        checks.check(f"fusion {method}", fused[1] == fused[2] and fused[1] > fused[0])
    family = pd.DataFrame([{"doc_id":"a","text":"a"},{"doc_id":"b","text":"b","original_doc_id":"a"},{"doc_id":"c","text":"c","original_doc_id":"b"},{"doc_id":"d","text":"d"}])
    first,second=grouped_split(documents=family, validation_fraction=.5)
    checks.check("translation chains never leak", {"a","b","c"} <= set(first.doc_id) or {"a","b","c"} <= set(second.doc_id))

    return windows


def masking(checks: Checks, tokenizer, windows):
    window = windows[0]
    count=len(window.words)
    signals=WindowSignals(entity=tuple(1.0 if index in (2,3) else 0.0 for index in range(count)), domain=tuple(float(index) for index in range(count)),keyphrase=tuple(float(count-index) for index in range(count)),uncertainty=tuple(float(index) for index in range(count)),entities=(Span(15,28,1,"DISEASE","C1"),))
    for identifier in range(9):
        policy=MaskPolicy(policy=f"M{identifier}", probability=.5, adaptive_enabled=identifier==8)
        selected=select_mask_positions(window=window,policy=policy,signals=signals,epoch=1)
        checks.equal(f"M{identifier} exact budget",len(selected.positions),math.floor(.5*len(window.eligible)))
        checks.check(f"M{identifier} valid targets",set(selected.positions)<=set(window.eligible))
        checks.equal(f"M{identifier} repeatable",selected,select_mask_positions(window=window,policy=policy,signals=signals,epoch=1))
        collator=MaskCollator(tokenizer=tokenizer,policy=policy,signals={(window.doc_id,window.index):signals})
        batch=collator([window]);labels=batch["labels"][0]
        checks.equal(f"M{identifier} labels count",int((labels!=-100).sum()),selected.budget)
        checks.check(f"M{identifier} special labels ignored",all(int(labels[position])==-100 for position in range(len(labels)) if position not in window.eligible))
    fragmented=Window("fragmented",0,(2,5,6,3),((0,0),(0,1),(1,2),(0,0)),(Word(0,2,"ab",(1,2)),),(1,2))
    checks.raises("impossible strict whole-word budget",ValueError,select_mask_positions,window=fragmented,policy=MaskPolicy(random_unit="word",probability=.5))
    checks.equal("explicit underfill deviation",select_mask_positions(window=fragmented,policy=MaskPolicy(random_unit="word",probability=.5,budget_mode="underfill")).deviation,-1)
    for fusion in ("hierarchical","disjoint","score","weighted_rank","rrf"):
        chosen=select_mask_positions(window=window,policy=MaskPolicy(policy="M7",probability=.5,fusion=fusion),signals=signals)
        checks.equal(f"{fusion} budget",len(chosen.positions),math.floor(.5*len(window.eligible)))
    nested=dataclasses.replace(signals,entities=(Span(15,28),Span(21,28)))
    chosen=select_mask_positions(window=window,policy=MaskPolicy(policy="M2",probability=.5),signals=nested)
    entity_positions={position for word in window.words[2:4] for position in word.positions}
    checks.check("nested entities indivisible",not (entity_positions&set(chosen.positions)) or entity_positions<=set(chosen.positions))
    collator=MaskCollator(tokenizer=tokenizer,policy=MaskPolicy(probability=1,mask_fraction=1,random_fraction=0))
    batch=collator([window])
    checks.check("all-mask replacement",all(batch['input_ids'][0,position]==tokenizer.mask_token_id for position in window.eligible))
    dynamic=MaskCollator(tokenizer=tokenizer,policy=MaskPolicy(probability=.5))
    a=dynamic([windows[0],windows[1]])
    b=dynamic([windows[1],windows[0]])
    checks.check("worker/order-independent masks",torch.equal(a['input_ids'][0],b['input_ids'][1]))
    dynamic.epoch=3;c=dynamic([windows[0],windows[1]])
    checks.check("epoch changes dynamic masks",not torch.equal(a['input_ids'],c['input_ids']))
    causal=MaskCollator(tokenizer=tokenizer,policy=MaskPolicy(),objective="clm")([window])
    checks.check("causal label shift",all(causal['labels'][0,position-1]==window.input_ids[position] for position in window.eligible if position>0))


def models_and_evaluation(checks: Checks, tokenizer, model, windows, root):
    for architecture in ("bert","roberta"):
        if architecture=="roberta":
            from transformers import RobertaConfig,RobertaForMaskedLM
            model=RobertaForMaskedLM(RobertaConfig(vocab_size=len(tokenizer),hidden_size=16,num_hidden_layers=1,num_attention_heads=2,intermediate_size=32,max_position_embeddings=64,pad_token_id=0))
        for mode in ("clm","mlm"):
            set_attention_mode(model=model,objective=mode)
            verify_attention(model=model,tokenizer=tokenizer,objective=mode)
            checks.check(f"{architecture} {mode} attention verified",True)
            batch=MaskCollator(tokenizer=tokenizer,policy=MaskPolicy(probability=.5),objective=mode)([windows[0]])
            logits=model(input_ids=batch['input_ids'],attention_mask=batch['attention_mask']).logits
            loss=torch.nn.functional.cross_entropy(logits.transpose(1,2),batch['labels'],ignore_index=-100)
            loss.backward();checks.check(f"{architecture} {mode} backward",all(torch.isfinite(parameter.grad).all() for parameter in model.parameters() if parameter.grad is not None));model.zero_grad()
        model.eval()
        first=evaluate_pretraining(model=model,tokenizer=tokenizer,windows=windows,probability=.5)
        second=evaluate_pretraining(model=model,tokenizer=tokenizer,windows=windows,probability=.5)
        checks.equal(f"{architecture} fixed evaluation",first,second)
        checks.check("no autoregressive MLM perplexity",'causal_ppl' not in first)
        pll=pseudo_likelihood(model=model,tokenizer=tokenizer,windows=windows[:1])
        checks.check("word PPPL definition",abs(math.log(pll['pppl'])+pll['mean_pll'])<1e-8)
        entity=pseudo_likelihood(model=model,tokenizer=tokenizer,windows=windows[:1],conditioning='whole_entity',entities={windows[0].doc_id:(Span(15,28),)})
        checks.equal("entity PLL token count",entity['targets'],2)
        uncertainty,cost=predictive_entropy(model=model,tokenizer=tokenizer,windows=windows[:1])
        checks.check("masked entropy cache and cost",len(next(iter(uncertainty.values())))==len(windows[0].words) and cost>0)
        path=root/f'roundtrip-{architecture}';model.save_pretrained(path);tokenizer.save_pretrained(path)
        loaded=type(model).from_pretrained(path)
        checks.check(f"{architecture} reload parameters exact",all(torch.equal(value,loaded.state_dict()[key]) for key,value in model.state_dict().items()))


def training(checks: Checks, root, base, train_df, dev_df, predictions):
    parameters=TrainingParameters(epochs=2,batch_size=1,gradient_accumulation=2,evaluation_probability=.5)
    common=dict(base_model=str(base),documents=train_df,validation=dev_df,training_parameters=parameters,max_length=32)
    clm=pretrain_causal(**common,output_dir=root/'clm')
    checks.equal("CLM accumulated optimizer steps",clm.global_step,4)
    shared=clm.final_checkpoint
    shared_hash=checkpoint_sha256(model_dir=shared)
    runs=[]
    for identifier in range(9):
        result=pretrain_masked(base_model=str(shared),documents=train_df,validation=dev_df,output_dir=root/f'M{identifier}',training_parameters=parameters,max_length=32,mask_policy=MaskPolicy(policy=f'M{identifier}',probability=.5,adaptive_enabled=identifier==8),entity_predictions=predictions,parent_sha256=shared_hash)
        runs.append(result)
        checks.equal(f"M{identifier} complete optimizer plan",result.global_step,4)
        metrics=pd.read_parquet(result.output_dir/'epoch_metrics.parquet')
        checks.equal(f"M{identifier} epoch zero included",sorted(metrics[metrics.group=='validation'].epoch.unique().tolist()),[0,1,2])
        checks.check(f"M{identifier} full artifacts",all((result.output_dir/name).exists() for name in ['run_manifest.json','resolved_config.yaml','events.jsonl','run.log','epoch_metrics.parquet','training_summary.json','checkpoints','evaluations']))
    checks.equal("shared parent unchanged",checkpoint_sha256(model_dir=shared),shared_hash)
    compared=compare_pretraining_runs(runs=[result.output_dir for result in runs[:8]])
    checks.equal("controlled branches compared",len(compared),8)
    paused=pretrain_causal(**common,output_dir=root/'paused',stop_after_epoch=1)
    resumed=pretrain_causal(**common,output_dir=root/'resumed',resume_from=paused.final_checkpoint.parent)
    original=BertForMaskedLM.from_pretrained(clm.final_checkpoint)
    restored=BertForMaskedLM.from_pretrained(resumed.final_checkpoint)
    checks.check("resume preserves exact CPU trajectory",all(torch.equal(value,restored.state_dict()[key]) for key,value in original.state_dict().items()))
    configuration={'task':'cpt.pretrain_masked','base_model':str(shared),'documents':str(root/'train.parquet'),'validation':str(root/'dev.parquet'),'output_dir':str(root/'yaml'),'mask_policy':{'probability':.5},'training_parameters':{'evaluation_probability':.5},'max_length':32}
    train_df.to_parquet(root/'train.parquet',index=False);dev_df.to_parquet(root/'dev.parquet',index=False)
    config_path=root/'smoke.yaml';config_path.write_text(yaml.safe_dump(configuration))
    executed=subprocess.run([sys.executable,'-m','lab.cli','run',str(config_path)],capture_output=True,text=True)
    checks.equal("existing YAML CLI executes CPT",executed.returncode,0)
    if executed.returncode: print(executed.stderr[-2000:])
    return runs



def extra_invariants(checks, root, tokenizer, model, windows, base, train_df, dev_df, predictions):
    import itertools
    from lab.cpt.masking.selection import select_budget_units
    from lab.cpt.sampling import match_corpus_volume,describe_corpus
    first,second,volume=match_corpus_volume(first=train_df,second=train_df.iloc[:2],tokenizer=tokenizer,token_budget=40,max_length=32)
    a=sum(len(window.eligible) for window in tokenize_corpus(documents=first,tokenizer=tokenizer,max_length=32))
    b=sum(len(window.eligible) for window in tokenize_corpus(documents=second,tokenizer=tokenizer,max_length=32))
    checks.equal('native/MT equal eligible volume',a,b)
    checks.equal('declared matched volume',a,volume)
    checks.equal('duplication statistics',describe_corpus(documents=train_df)['duplicate_text_documents'],2)
    for sizes in [(1,2,3),(2,2),(3,5),(2,3,4)]:
        units=[];position=0
        for size in sizes:
            units.append(tuple(range(position,position+size)));position+=size
        feasible={sum(size for size,flag in zip(sizes,flags) if flag) for flags in itertools.product((0,1),repeat=len(sizes))}
        for budget in range(sum(sizes)+1):
            if budget in feasible:
                checks.equal(f'exact subset {sizes}/{budget}',sum(map(len,select_budget_units(units,budget,'strict'))),budget)
            else:
                checks.raises(f'impossible subset {sizes}/{budget}',ValueError,select_budget_units,units,budget,'strict')
    uniform=BertForMaskedLM(model.config)
    for parameter in uniform.parameters(): parameter.data.zero_()
    set_attention_mode(model=uniform,objective='mlm')
    metrics=evaluate_pretraining(model=uniform,tokenizer=tokenizer,windows=windows,probability=.5)
    checks.check('fixed-mask CE equals log vocabulary under uniform logits',abs(metrics['nll']-math.log(len(tokenizer)))<1e-6)
    pll=pseudo_likelihood(model=uniform,tokenizer=tokenizer,windows=windows[:1])
    checks.check('uniform word PPPL equals vocabulary size',abs(pll['pppl']-len(tokenizer))<1e-5)
    from lab.cpt.evaluation.checkpoints import evaluate_checkpoint
    original=evaluate_checkpoint(base_model=str(base),documents=dev_df,probability=.5)
    checks.check('original checkpoint evaluated',bool(original.metrics))
    parameters=TrainingParameters(epochs=2,batch_size=1,gradient_accumulation=2,evaluation_probability=.5)
    for identifier in (0,8):
        policy=MaskPolicy(policy=f'M{identifier}',probability=.5,adaptive_enabled=identifier==8,adaptive_refresh_epochs=2)
        common=dict(base_model=str(base),documents=train_df,validation=dev_df,training_parameters=parameters,max_length=32,mask_policy=policy)
        full=pretrain_masked(**common,output_dir=root/f'resume-full-{identifier}')
        partial=pretrain_masked(**common,output_dir=root/f'resume-part-{identifier}',stop_after_epoch=1)
        resumed=pretrain_masked(**common,output_dir=root/f'resume-rest-{identifier}',resume_from=partial.final_checkpoint.parent)
        left=BertForMaskedLM.from_pretrained(full.final_checkpoint)
        right=BertForMaskedLM.from_pretrained(resumed.final_checkpoint)
        checks.check(f'M{identifier} exact resume trajectory',all(torch.equal(value,right.state_dict()[key]) for key,value in left.state_dict().items()))
        checks.check('resumed best checkpoint exists',resumed.best_checkpoint.is_dir())
        if identifier==8:
            metrics=pd.read_parquet(full.output_dir/'epoch_metrics.parquet')
            costs=metrics[metrics.metric=='adaptive_inference_seconds'].value.tolist()
            checks.check('entropy refresh frequency reuses valid cache',costs[0]>0 and costs[1]==0)
    sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'examples'))
    from cpt_downstream import frozen_linking_probe,exact_ner_metrics
    set_attention_mode(model=model,objective='mlm')
    terminology=pd.DataFrame([{'term':'heart failure','code':'C1'},{'term':'pulmonary edema','code':'C2'},{'term':'heart failure','code':'C1'}])
    mentions=pd.DataFrame([{'text':'heart failure','code':'C1','context':TEXT,'start_span':15,'end_span':28}])
    for mode in ('mention_only','contextual'):
        metrics=frozen_linking_probe(model=model,tokenizer=tokenizer,terminology=terminology,mentions=mentions,mode=mode,seen_concepts=frozenset({'C1'}))
        checks.equal(f'{mode} deduplicated candidates',metrics['mean_candidates'],2)
        checks.check(f'{mode} valid retrieval metrics',all(0<=metrics[key]<=1 for key in ('recall@1','recall@5','recall@25','mrr')))
    spans=predictions.iloc[:1]
    scores=exact_ner_metrics(predicted=spans,gold=spans)
    strict=scores['strict']
    checks.check('canonical exact NER F1',strict.f1==1)
    for run_dir in (root/'M3',root/'M7'):
        manifest=(run_dir/'run_manifest.json').read_text()
        checks.check('manifest has no raw corpus or lexical tokens',TEXT not in manifest and 'heart' not in manifest)

def main():
    checks=Checks('CPT offline')
    root=Path(os.environ.get('CPT_VERIFY_OUTPUT',tempfile.mkdtemp(prefix='lab-cpt-')))
    root.mkdir(parents=True,exist_ok=True)
    tokenizer,model,base,training_df,validation,predictions=fixture(root)
    windows=contracts(checks,tokenizer,training_df,predictions)
    masking(checks,tokenizer,windows)
    models_and_evaluation(checks,tokenizer,model,windows,root)
    checks.check("CPT tasks registered",any(name=='cpt.pretrain_causal' for name,_ in list_tasks()))
    checks.equal("task resolves public API",resolve_task('cpt.pretrain_causal'),pretrain_causal)
    training(checks,root,base,training_df,validation,predictions)
    extra_invariants(checks,root,tokenizer,model,windows,base,training_df,validation,predictions)
    print(f'Artifacts: {root}')
    return checks.report()


if __name__=='__main__':
    run(main)
