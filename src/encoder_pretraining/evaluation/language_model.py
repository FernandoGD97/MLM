"""Target-weighted intrinsic evaluation and long-text pseudo-log-likelihood."""
import json,math,random
from pathlib import Path

def evaluate_labeled_batches(model,loader,objective,device):
    import torch
    was_training=model.training; model.eval(); nll_sum=targets=correct=top5=0
    with torch.inference_mode():
        for batch in loader:
            batch={k:v.to(device) for k,v in batch.items() if hasattr(v,"to")}; labels=batch["labels"]; valid=labels!=-100
            logits=model(**batch).logits
            losses=torch.nn.functional.cross_entropy(logits.transpose(1,2),labels,ignore_index=-100,reduction="sum")
            nll_sum+=float(losses); targets+=int(valid.sum()); pred=logits.argmax(-1)
            correct+=int(((pred==labels)&valid).sum()); top5+=int(((logits.topk(min(5,logits.size(-1)),-1).indices==labels.unsqueeze(-1)).any(-1)&valid).sum())
    model.train(was_training); nll=nll_sum/targets if targets else float("nan")
    if objective=="clm": return {"validation_clm_nll":nll,"validation_clm_ppl":math.exp(min(nll,50)),"clm_prediction_targets":targets}
    return {"mlm_validation_nll":nll,"masked_token_accuracy":correct/max(1,targets),"masked_top5_accuracy":top5/max(1,targets),"mlm_prediction_targets":targets}

class MiniconsStridedBackend:
    """Exact masked PLL with bounded-context windows and `within_word_l2r` grouping.

    Each token is scored once. Long inputs produce additional windows rather than silently
    dropping suffixes. Later subwords of the target word remain masked for Kauf-Ivanova's
    within-word left-to-right conditioning.
    """
    def __init__(self,tokenizer,window_size=128,stride=64,model_batch_size=32,pll_method="within_word_l2r"):
        if pll_method!="within_word_l2r": raise ValueError("Only scientifically preferred within_word_l2r is supported")
        if stride<=0 or window_size<=2: raise ValueError("Invalid window/stride")
        self.tok,self.window,self.stride,self.batch,self.method=tokenizer,window_size,stride,model_batch_size,pll_method
    def score(self,model,texts,device,spans=None):
        import torch
        total_pll=0.; token_count=0; document_scores=[]; was_training=model.training; model.eval()
        with torch.inference_mode():
            for document_index,text in enumerate(texts):
                enc=self.tok(text,add_special_tokens=False,return_offsets_mapping=True); ids=enc["input_ids"]; offsets=enc["offset_mapping"]
                variants=[]; positions=[]; gold=[]
                targets=range(len(ids))
                if spans is not None:
                    a,b=spans[document_index]; targets=[i for i,(x,y) in enumerate(offsets) if y>x and x<b and y>a]
                for target in targets:
                    start=min((target//self.stride)*self.stride,max(0,len(ids)-(self.window-2))); end=min(len(ids),start+self.window-2)
                    local=list(ids[start:end]); previous_end=offsets[target][1]
                    for j in range(target+1,end):
                        if any(c.isspace() for c in text[previous_end:offsets[j][0]]): break
                        local[j-start]=self.tok.mask_token_id
                        previous_end=offsets[j][1]
                    local[target-start]=self.tok.mask_token_id
                    prepared=self.tok.prepare_for_model(local,add_special_tokens=True,return_special_tokens_mask=True)
                    prefix=next(i for i,is_special in enumerate(prepared["special_tokens_mask"]) if not is_special)
                    variants.append(prepared["input_ids"]); positions.append(prefix+target-start); gold.append(ids[target])
                doc_pll=0.
                for p in range(0,len(variants),self.batch):
                    chunk=variants[p:p+self.batch]; width=max(map(len,chunk)); pad=self.tok.pad_token_id
                    x=torch.tensor([v+[pad]*(width-len(v)) for v in chunk],device=device); mask=x.ne(pad).long()
                    logits=model(input_ids=x,attention_mask=mask).logits
                    rows=torch.arange(len(chunk),device=device); pos=torch.tensor(positions[p:p+self.batch],device=device); target=torch.tensor(gold[p:p+self.batch],device=device)
                    doc_pll+=float(torch.log_softmax(logits[rows,pos],-1)[rows,target].sum())
                document_scores.append(doc_pll); total_pll+=doc_pll; token_count+=len(targets)
        model.train(was_training)
        mean_pll=total_pll/max(1,token_count)
        return {"minicons_pppl":math.exp(min(-mean_pll,50)),"minicons_mean_pll":mean_pll,
                "n_documents_scored":len(document_scores),"n_tokens_scored":token_count},document_scores

class SpanMiniconsStridedBackend(MiniconsStridedBackend):
    def score_spans(self,model,texts,spans,device): return self.score(model,texts,device,spans=spans)

def fixed_pppl_subset(documents,max_documents,seed,manifest_path):
    path=Path(manifest_path)
    if path.exists(): indices=json.loads(path.read_text())["indices"]
    else:
        indices=list(range(len(documents))); random.Random(seed).shuffle(indices); indices=sorted(indices[:max_documents])
        path.parent.mkdir(parents=True,exist_ok=True); path.write_text(json.dumps({"seed":seed,"indices":indices},indent=2))
    return [documents[i] for i in indices]
