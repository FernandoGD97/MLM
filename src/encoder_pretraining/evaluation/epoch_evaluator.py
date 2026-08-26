"""Epoch-zero/epoch-end intrinsic and downstream evaluation coordinator."""
from pathlib import Path
import json
from .language_model import evaluate_labeled_batches,MiniconsStridedBackend,fixed_pppl_subset
from .entity_linking.evaluator import EntityLinkingEvaluator
from ..training.model_mode import bidirectional_evaluation

def _schedule(config):
    from ..masking import FixedSchedule,LinearSchedule,CosineSchedule
    s=config.get("schedule",config.get("mask_schedule",{"type":"fixed","probability":config.get("probability",.15)}))
    return {"fixed":FixedSchedule,"linear":LinearSchedule,"cosine":CosineSchedule}[s["type"]](**{k:v for k,v in s.items() if k!="type"})
def build_training_collator(config,tokenizer):
    from ..collators import MLMCollator,CLMCollator
    from ..masking import RandomMasking,EntityFirstMasking
    from ..entities import NoneEntityProvider,GazetteerEntityProvider,HuggingFaceNERProvider
    if config["objective"]=="clm": return CLMCollator(tokenizer)
    m=config["masking"]; policy=RandomMasking() if m.get("strategy","random")=="random" else EntityFirstMasking(m.get("entity_budget_fraction",.5),m.get("remainder","random"),m.get("stopwords",[]))
    e=config.get("entities",{"provider":"none"})
    if e["provider"]=="none":provider=NoneEntityProvider()
    elif e["provider"]=="gazetteer":provider=GazetteerEntityProvider(Path(e["path"]).read_text().splitlines())
    elif e["provider"]=="hf_ner":provider=HuggingFaceNERProvider(e["checkpoint"])
    else:raise ValueError(e["provider"])
    return MLMCollator(tokenizer,_schedule(m),policy,provider,m.get("replacement","bert_80_10_10"),int(config["training"].get("seed",42)))

class EpochEvaluator:
    def __init__(self,config,validation,tracker):
        self.config,self.validation,self.tracker=config,validation,tracker; self.objective=config["objective"]
        self.el=EntityLinkingEvaluator(config.get("evaluation",{}).get("entity_linking",{})) if config.get("evaluation",{}).get("entity_linking",{}).get("enabled") else None
        self.fixed_collator=None
    def evaluate(self,model,tokenizer,checkpoint_dir,epoch,global_step,processed_tokens):
        import torch
        cfg=self.config.get("evaluation",{}); every=int(cfg.get("every_n_epochs",1)); final=epoch==int(self.config["training"]["epochs"])
        if epoch and epoch%every and not final:return {}
        checkpoint_dir=Path(checkpoint_dir); evaluation_dir=checkpoint_dir/"evaluation"; evaluation_dir.mkdir(parents=True,exist_ok=True)
        device=next(model.parameters()).device; all_metrics={}; was_training=model.training
        lm_cfg=cfg.get("language_model",{})
        if lm_cfg.get("enabled",True) and self.validation:
            from torch.utils.data import DataLoader
            if self.objective=="mlm":
                if self.fixed_collator is None:self.fixed_collator=build_training_collator(self.config,tokenizer)
                self.fixed_collator.reset_seed()
                loader=DataLoader(self.validation,batch_size=int(lm_cfg.get("batch_size",8)),shuffle=False,collate_fn=self.fixed_collator)
            else:
                from ..collators import CLMCollator; loader=DataLoader(self.validation,batch_size=int(lm_cfg.get("batch_size",8)),shuffle=False,collate_fn=CLMCollator(tokenizer))
            metrics=evaluate_labeled_batches(model,loader,self.objective,device); all_metrics.update(metrics)
            self.tracker.add_metrics(epoch,global_step,processed_tokens,self.objective,"language_model",lm_cfg.get("dataset_name","validation"),metrics)
        def bidirectional_diagnostics():
            pppl=lm_cfg.get("pppl",{})
            if pppl.get("enabled") and self.validation:
                documents=[]; seen=set()
                for x in self.validation:
                    key=str(x.get("doc_id"));
                    if key not in seen:documents.append(x["text"]);seen.add(key)
                subset=fixed_pppl_subset(documents,int(pppl.get("max_documents",500)),int(self.config["training"].get("seed",42)),self.tracker.run_dir/"pppl_eval_manifest.json")
                backend=MiniconsStridedBackend(tokenizer,int(pppl.get("window_size",128)),int(pppl.get("stride",64)),int(pppl.get("model_batch_size",32)),pppl.get("pll_method","within_word_l2r"))
                metrics,_=backend.score(model,subset,device); all_metrics.update(metrics)
                self.tracker.add_metrics(epoch,global_step,processed_tokens,self.objective,"language_model",pppl.get("dataset_name","pppl"),metrics)
            if self.el and (epoch%int(self.el.config.get("every_n_epochs",1))==0 or final):
                results=self.el.evaluate(model,tokenizer,checkpoint_dir,epoch,final)
                for name,metrics in results.items():self.tracker.add_metrics(epoch,global_step,processed_tokens,self.objective,"entity_linking",name,metrics)
        if self.objective=="clm":
            with bidirectional_evaluation(model):bidirectional_diagnostics()
        else:bidirectional_diagnostics()
        model.train(was_training); (evaluation_dir/"lm_metrics.json").write_text(json.dumps(all_metrics,indent=2,sort_keys=True))
        return all_metrics
