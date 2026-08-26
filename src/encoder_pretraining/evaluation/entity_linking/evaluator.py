"""Zero-shot concept retrieval; terminology indexes are built once per epoch."""
from pathlib import Path
import json
from ..base import DownstreamEvaluator
from .datasets import load_terminology,load_mentions
from .encoding import encode_texts
from .candidates import CandidateIndex
from .metrics import compute_metrics

class EntityLinkingEvaluator(DownstreamEvaluator):
    def __init__(self,config): self.config=config; self.index_builds={}
    def evaluate(self,model,tokenizer,output_dir,epoch,final=False):
        import torch
        cfg=self.config; datasets=[d for d in cfg.get("datasets",[]) if d.get("role","monitor")=="monitor" or final]
        device=next(model.parameters()).device; indexes={}; results={}; out=Path(output_dir)/"evaluation"/"entity_linking"
        for terminology in sorted({d["terminology"] for d in datasets}):
            terms=load_terminology(cfg["terminologies"][terminology]); ids,texts=zip(*terms)
            embeddings=encode_texts(model,tokenizer,list(texts),device,cfg.get("batch_size",32),cfg.get("pooling","mean"),cfg.get("normalize_embeddings",True),cfg.get("max_length",128))
            indexes[terminology]=CandidateIndex(embeddings,list(ids),list(texts),cfg.get("normalize_embeddings",True)); self.index_builds[terminology]=self.index_builds.get(terminology,0)+1
        for dataset in datasets:
            mentions=load_mentions(dataset); query_builder=dataset.get("query_builder","mention_only")
            if query_builder=="mention_only": queries=[m.mention for m in mentions]
            elif query_builder=="mention_with_context": queries=[m.text if m.text is not None else m.mention for m in mentions]
            else: raise ValueError(query_builder)
            q=encode_texts(model,tokenizer,queries,device,cfg.get("batch_size",32),cfg.get("pooling","mean"),cfg.get("normalize_embeddings",True),cfg.get("max_length",128))
            retrieved=indexes[dataset["terminology"]].search(q,max(cfg["top_k"])); metrics,ranks=compute_metrics([m.gold_concept_id for m in mentions],retrieved,cfg["top_k"])
            span_cfg=dataset.get("span_pppl",{})
            span_mentions=[m for m in mentions if m.text is not None and m.start is not None and m.end is not None]
            if span_cfg.get("enabled") and span_mentions:
                import math,statistics
                from ..language_model import SpanMiniconsStridedBackend
                backend=SpanMiniconsStridedBackend(tokenizer,int(span_cfg.get("window_size",128)),int(span_cfg.get("stride",64)),int(span_cfg.get("model_batch_size",32)),span_cfg.get("pll_method","within_word_l2r"))
                _,pll=backend.score_spans(model,[m.text for m in span_mentions],[(m.start,m.end) for m in span_mentions],device)
                nlls=[]
                for mention,score in zip(span_mentions,pll):
                    offsets=tokenizer(mention.text,add_special_tokens=False,return_offsets_mapping=True)["offset_mapping"]
                    count=sum(y>x and x<mention.end and y>mention.start for x,y in offsets)
                    if count:nlls.append(-score/count)
                pppls=[math.exp(min(x,50)) for x in nlls]
                metrics.update(entity_span_pppl_geomean=math.exp(sum(nlls)/len(nlls)) if nlls else float("nan"),
                               entity_span_pppl_median=statistics.median(pppls) if pppls else float("nan"),n_entities_scored=len(nlls))
            target=out/dataset["name"]; target.mkdir(parents=True,exist_ok=True); (target/"metrics.json").write_text(json.dumps(metrics,indent=2,sort_keys=True))
            if dataset.get("save_predictions",True):
                import pyarrow as pa,pyarrow.parquet as pq
                rows=[]
                for m,row,rank in zip(mentions,retrieved,ranks):
                    gold_hit=next((x for x in row if x[0]==m.gold_concept_id),None); top=row[0] if row else (None,None,None)
                    rows.append({"doc_id":m.doc_id,"mention":m.mention,"gold_concept_id":m.gold_concept_id,"gold_rank":rank,
                                 "top1_concept":top[0],"top1_score":top[1],"gold_score":gold_hit[1] if gold_hit else None})
                pq.write_table(pa.Table.from_pylist(rows),target/"per_mention_results.parquet")
            results[dataset["name"]]=metrics
        del indexes
        if torch.cuda.is_available(): torch.cuda.empty_cache()
        return results
