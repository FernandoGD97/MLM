from dataclasses import dataclass
@dataclass
class CandidateIndex:
    embeddings: object; concept_ids:list[str]; terms:list[str]; normalized:bool=True
    def search(self,queries,k):
        import torch
        if self.normalized: queries=torch.nn.functional.normalize(queries,p=2,dim=-1)
        scores=queries@self.embeddings.T; alias_k=min(scores.size(1),max(k*8,k)); values,indices=scores.topk(alias_k,-1)
        results=[]
        for vals,inds in zip(values.tolist(),indices.tolist()):
            seen=set(); row=[]
            for score,index in zip(vals,inds):
                concept=self.concept_ids[index]
                if concept not in seen: row.append((concept,float(score),self.terms[index])); seen.add(concept)
                if len(row)>=k: break
            # If many aliases exhaust the heuristic alias_k, rank all aliases exactly.
            if len(row)<min(k,len(set(self.concept_ids))):
                vals,inds=scores[len(results)].sort(descending=True); seen=set(); row=[]
                for score,index in zip(vals.tolist(),inds.tolist()):
                    concept=self.concept_ids[index]
                    if concept not in seen: row.append((concept,float(score),self.terms[index])); seen.add(concept)
                    if len(row)>=k: break
            results.append(row)
        return results
