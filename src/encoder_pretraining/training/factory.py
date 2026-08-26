"""Configuration factory reused by both public entrypoints."""
from pathlib import Path
import random

def run_from_config(path, required_objective):
    import yaml, torch
    from transformers import AutoTokenizer, AutoModelForMaskedLM
    from ..data import load_parquet, tokenize_and_chunk
    from ..objectives import set_attention_mode
    from ..tracking.metrics import MetricsTracker
    from ..tracking.manifest import write_run_manifest
    from ..evaluation.epoch_evaluator import EpochEvaluator
    from .engine import TrainingEngine
    cfg=yaml.safe_load(Path(path).read_text()); objective=cfg.get("objective")
    if objective != required_objective: raise ValueError(f"Expected objective={required_objective}, got {objective}")
    seed=int(cfg["training"].get("seed",42)); random.seed(seed); torch.manual_seed(seed)
    tok=AutoTokenizer.from_pretrained(cfg["model"]["checkpoint"],use_fast=True)
    resume=cfg["training"].get("resume_from_checkpoint")
    model=AutoModelForMaskedLM.from_pretrained(str(Path(resume)/"model") if resume else cfg["model"]["checkpoint"]); set_attention_mode(model,objective)
    train_docs=load_parquet(**cfg["data"]["train"]); train_data,stats=tokenize_and_chunk(train_docs,tok,**cfg["chunking"])
    validation=[]
    if cfg["data"].get("validation"):
        validation,_=tokenize_and_chunk(load_parquet(**cfg["data"]["validation"]),tok,**cfg["chunking"])
    from ..evaluation.epoch_evaluator import build_training_collator
    collator=build_training_collator(cfg,tok)
    run_dir=Path(cfg["training"]["run_dir"]); run_dir.mkdir(parents=True,exist_ok=True); (run_dir/"config.yaml").write_text(yaml.safe_dump(cfg,sort_keys=False))
    write_run_manifest(run_dir,cfg,tok,stats)
    tracker=MetricsTracker(run_dir)
    evaluator=EpochEvaluator(cfg,validation,tracker)
    engine_cfg={**cfg["training"],"run_before_training":cfg.get("evaluation",{}).get("run_before_training",True)}
    return TrainingEngine(model,tok,train_data,collator,objective,engine_cfg,evaluator,tracker).run()
