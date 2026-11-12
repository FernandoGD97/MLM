"""Offline term-source and genuine Transformer-attention inspection checks."""
from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path

import pandas as pd
import torch

from _harness import Checks, run
from verify_cpt import fixture
from lab.cpt.inspection.attention import attention_signals, incoming_attention
from lab.cpt.inspection.corpus import read_corpora, segment_corpora
from lab.cpt.inspection.signals import term_spans
from lab.cpt.inspection.terms import read_terms
from lab.cpt.inspection.workflow import inspect_corpora


def main():
    checks=Checks('Term lists and attention')
    root=Path(tempfile.mkdtemp(prefix='term-attention-'))
    txt=root/'terms.txt';txt.write_text('\ufeffheart failure\n\npulmonary edema\nheart failure\n',encoding='utf-8')
    parquet=root/'terms.parquet';pd.DataFrame({'terms':['pulmonary edema','without']}).to_parquet(parquet,index=False)
    checks.equal('phrases and deduplication',read_terms(paths=[txt,parquet]),('heart failure','pulmonary edema','without'))
    checks.equal('literal multiword match',[(span.start,span.end) for span in term_spans(text='Heart failure, heart failures.',terms=['heart failure'],case_sensitive=False)],[(0,13)])
    for name,frame in [('wrong',pd.DataFrame({'term':['a']})),('extra',pd.DataFrame({'terms':['a'],'score':[1]})),('null',pd.DataFrame({'terms':[None]})),('numeric',pd.DataFrame({'terms':[123]})),('blank',pd.DataFrame({'terms':[' ']}))]:
        path=root/f'{name}.parquet';frame.to_parquet(path,index=False)
        checks.raises(f'invalid {name} rejected',ValueError,read_terms,paths=[path])
    empty=root/'empty.txt';empty.write_text('\n')
    checks.raises('empty list rejected',ValueError,read_terms,paths=[empty])
    matrix=torch.tensor([[[[.1,.2,.7]]*3]])
    scores=incoming_attention(attentions=(matrix,),eligible=(0,1,2),word_positions=((0,),(1,2)))
    checks.check('received attention and mean subwords exact',abs(scores[0]-.3)<1e-6 and abs(scores[1]-1.35)<1e-6)
    special=incoming_attention(attentions=(torch.tensor([[[[.8,.05,.15]]*3]]),),eligible=(1,2),word_positions=((1,),(2,)))
    checks.check('special tokens excluded and relative normalization',abs(special[0]-.5)<1e-6 and abs(special[1]-1.5)<1e-6)
    uniform=incoming_attention(attentions=(torch.ones(1,2,3,3)/3,),eligible=(0,1,2),word_positions=((0,),(1,2)))
    checks.equal('uniform attention does not reward fragmented word',uniform,(1.,1.))
    checks.raises('missing attention fails',ValueError,incoming_attention,attentions=(),eligible=(0,),word_positions=((0,),))
    checks.raises('nonfinite attention fails',ValueError,incoming_attention,attentions=(torch.full((1,1,1,1),float('nan')),),eligible=(0,),word_positions=((0,),))
    _,_,base,training,_,_=fixture(root/'fixture')
    corpus=root/'corpus.parquet';training.to_parquet(corpus,index=False)
    docs=segment_corpora(frames=read_corpora(paths=[corpus]),language='en')
    direct=attention_signals(documents=docs,base_model=str(base),max_length=8)
    again=attention_signals(documents=docs,base_model=str(base),max_length=8)
    checks.equal('eval mode attention repeatable',direct.scores,again.scores)
    checks.check('long document windows aligned completely',all(len(scores)==len(doc.words) for scores,doc in zip(direct.scores,docs)))
    checks.raises('invalid layer count',ValueError,attention_signals,documents=docs,base_model=str(base),layers=2)
    result=inspect_corpora(corpora=[str(corpus)],output_dir=str(root/'api'),language='en',techniques=['terms','attention'],term_files=[str(txt),str(parquet)],probability=.2,base_model=str(base),attention_max_length=8)
    scores_df=pd.read_parquet(result.output_dir/'attention_scores.parquet')
    checks.equal('all word scores persisted',len(scores_df),sum(len(doc.words) for doc in docs))
    checks.check('no raw text in attention score artifact','text' not in scores_df)
    metadata=json.loads((result.output_dir/'attention_metadata.json').read_text())
    checks.check('checkpoint identity and measured cost',len(metadata['checkpoint_sha256'])==64 and metadata['seconds']>0)
    for technique in ['terms','attention']:
        summary=json.loads((result.output_dir/technique/'summary.json').read_text())
        checks.equal(f'{technique} global budget',summary['selected_words'],int(.2*sum(len(doc.words) for doc in docs)))
    command=[sys.executable,'scripts/inspect_corpus.py',str(corpus),'--language','en','--output-dir',str(root/'cli'),'--terms-file',str(txt),str(parquet),'--techniques','terms','attention','--base-model',str(base),'--probability','.2']
    process=subprocess.run(command,capture_output=True,text=True)
    checks.equal('driver TXT+Parquet+attention through CLI',process.returncode,0)
    if process.returncode: print(process.stderr)
    checks.check('driver persists scores',(root/'cli/masking/attention_scores.parquet').exists())
    alias=[sys.executable,'scripts/inspect_corpus.py',str(corpus),'--language','en','--output-dir',str(root/'alias'),'--terms',str(parquet),'--techniques','terms']
    process=subprocess.run(alias,capture_output=True,text=True)
    checks.equal('terms accepts an existing list file',process.returncode,0)
    print('Artifacts:',root)
    return checks.report()


if __name__=='__main__': run(main)
