#!/usr/bin/env python
"""Configuration-only entry point for continual pretraining."""
import argparse, json, random
from pathlib import Path

def main():
    import yaml, torch
    from transformers import AutoTokenizer, AutoModelForMaskedLM, Trainer, TrainingArguments
    from encoder_pretraining.data import load_parquet, tokenize_and_chunk
    from encoder_pretraining.masking import FixedSchedule, LinearSchedule, CosineSchedule, RandomMasking, EntityFirstMasking
    from encoder_pretraining.entities import NoneEntityProvider, GazetteerEntityProvider, HuggingFaceNERProvider
    from encoder_pretraining.collators import MLMCollator, CLMCollator
    from encoder_pretraining.objectives import set_attention_mode
    p=argparse.ArgumentParser(); p.add_argument("--config",required=True); args=p.parse_args()
    cfg=yaml.safe_load(Path(args.config).read_text()); seed=cfg.get("seed",42)
    random.seed(seed); torch.manual_seed(seed)
    tokenizer=AutoTokenizer.from_pretrained(cfg["model"]["checkpoint"],use_fast=True)
    model=AutoModelForMaskedLM.from_pretrained(cfg["model"]["checkpoint"])
    docs=load_parquet(**cfg["data"]); seqs,stats=tokenize_and_chunk(docs,tokenizer,**cfg["chunking"])
    print(json.dumps(stats.to_dict(),indent=2))
    phases=cfg.get("phases",[{"objective":cfg["objective"],**cfg}])
    for index,phase in enumerate(phases):
        objective=phase["objective"]; set_attention_mode(model,objective)
        if index:
            model.save_pretrained(Path(cfg["training"]["output_dir"])/f"phase-{index}-boundary")
        if objective=="clm": collator=CLMCollator(tokenizer)
        else:
            m=phase.get("masking",cfg.get("masking",{})); s=m.get("mask_schedule",{"type":"fixed","probability":m.get("probability",.15)})
            schedule={"fixed":FixedSchedule,"linear":LinearSchedule,"cosine":CosineSchedule}[s["type"]](**{k:v for k,v in s.items() if k!="type"})
            policy=RandomMasking() if m.get("strategy","random")=="random" else EntityFirstMasking(m.get("entity_budget_fraction",.5),m.get("remainder","random"),m.get("stopwords",[]))
            e=phase.get("entities",cfg.get("entities",{"provider":"none"})); provider={"none":lambda:NoneEntityProvider(),"gazetteer":lambda:GazetteerEntityProvider(Path(e["path"]).read_text().splitlines()),"hf_ner":lambda:HuggingFaceNERProvider(e["checkpoint"])}[e["provider"]]()
            collator=MLMCollator(tokenizer,schedule,policy,provider,m.get("replacement","bert_80_10_10"),seed)
        # Sequences retain text/offsets for dynamic entity projection; remove_unused_columns must remain false.
        ta={**cfg["training"],"remove_unused_columns":False,"seed":seed,"data_seed":seed}
        if phase.get("steps") is not None: ta["max_steps"]=phase["steps"]
        elif phase.get("tokens") is not None:
            # Token budgets, rather than epochs, are the primary compute-matching unit.
            import math
            mean_tokens=max(1,stats.tokens_per_sequence)
            ta["max_steps"]=math.ceil(phase["tokens"]/(ta["per_device_train_batch_size"]*mean_tokens))
        trainer=Trainer(model=model,args=TrainingArguments(**ta),train_dataset=seqs,data_collator=collator)
        trainer.train()
    model.config.is_decoder=False; model.save_pretrained(cfg["training"]["output_dir"]); tokenizer.save_pretrained(cfg["training"]["output_dir"])
if __name__=="__main__": main()
