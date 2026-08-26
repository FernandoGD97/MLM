import statistics
def compute_metrics(gold_ids,retrieved,top_k):
    ranks=[]
    for gold,row in zip(gold_ids,retrieved):
        ids=[x[0] for x in row]; ranks.append(ids.index(str(gold))+1 if str(gold) in ids else None)
    valid=[x for x in ranks if x is not None]; n=len(ranks)
    metrics={f"recall@{k}":sum(r is not None and r<=k for r in ranks)/max(1,n) for k in top_k}
    metrics.update(MRR=sum(1/r for r in valid)/max(1,n),mean_rank=statistics.mean(valid) if valid else float("nan"),
                   median_rank=statistics.median(valid) if valid else float("nan"),n_mentions=n,gold_not_found=n-len(valid))
    return metrics,ranks
