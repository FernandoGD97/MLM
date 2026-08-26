#!/usr/bin/env python
import argparse
from pathlib import Path
import yaml
from encoder_pretraining.data import load_parquet, tokenize_and_chunk
from encoder_pretraining.masking import FixedSchedule, RandomMasking, EntityFirstMasking
from encoder_pretraining.collators import MLMCollator
from encoder_pretraining.entities import NoneEntityProvider, GazetteerEntityProvider

def main():
    from transformers import AutoTokenizer
    p=argparse.ArgumentParser(); p.add_argument("--config",required=True); p.add_argument("--examples",type=int,default=20); a=p.parse_args()
    c=yaml.safe_load(Path(a.config).read_text()); tok=AutoTokenizer.from_pretrained(c["model"]["checkpoint"],use_fast=True)
    seqs,_=tokenize_and_chunk(load_parquet(**c["data"]),tok,**c["chunking"]); m=c.get("masking",c["phases"][-1]["masking"])
    policy=RandomMasking() if m["strategy"]=="random" else EntityFirstMasking(m.get("entity_budget_fraction",.5),m.get("remainder","random"),m.get("stopwords",[]))
    ec=c.get("entities",{"provider":"none"}); provider=NoneEntityProvider() if ec["provider"]=="none" else GazetteerEntityProvider(Path(ec["path"]).read_text().splitlines())
    col=MLMCollator(tok,FixedSchedule(m.get("probability",m.get("mask_schedule",{}).get("probability",.15))),policy,provider,m.get("replacement","bert_80_10_10"),c.get("seed",42),include_metadata=True)
    for ex in seqs[:a.examples]:
        out=col([ex]); meta=out["masking_metadata"][0]
        print("ORIGINAL:\n",tok.decode(ex["input_ids"],skip_special_tokens=True)); print("MASKED:\n",tok.decode(out["input_ids"][0],skip_special_tokens=False))
        print("ORIGINS:",sorted(meta.origins.items()),"\n")
if __name__=="__main__":main()
