"""Offline checks for canonical Parquets, multilingual segmentation and proportional masking."""
from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path

import pandas as pd

from _harness import Checks, run
from lab.core.sentences import ISO_639_1_LANGUAGES
from lab.cpt.inspection.corpus import read_corpora, segment_corpora
from lab.cpt.inspection.selection import proportional_budgets
from lab.cpt.inspection.workflow import inspect_corpora


def main():
    checks=Checks('Parquet inspection')
    root=Path(tempfile.mkdtemp(prefix='inspection-'))
    first=pd.DataFrame([{'doc_id':'same-id','text':'El paciente tiene dolor. No tiene fiebre.'}])
    second=pd.DataFrame([{'doc_id':f'doc-{i}','text':'The patient has heart failure. No edema today.'} for i in range(3)])
    a=root/'a.parquet';b=root/'b.parquet'
    first.to_parquet(a,index=False);second.to_parquet(b,index=False)
    frames=read_corpora(paths=[a,b])
    checks.equal('unannotated input accepted',frames[0].n_entities.tolist(),[0])
    docs=segment_corpora(frames=frames,languages=['es','en'])
    checks.equal('document counts preserved',len(docs),4)
    checks.check('sentence segmentation',all(len(doc.sentences)==2 for doc in docs))
    for code in sorted(ISO_639_1_LANGUAGES):
        doc=segment_corpora(frames=frames[:1],language=code)[0]
        checks.check(f'language {code}',bool(doc.sentences) and all(0<=left<right<=len(doc.text) for left,right in doc.sentences))
    checks.raises('language required',ValueError,segment_corpora,frames=frames)
    checks.equal('proportional ordinary budget',proportional_budgets(pools=[[(i,) for i in range(20)],[(i,) for i in range(80)]],quotas=[4,16],budget=20,mode='strict'),(4,16))
    checks.equal('constrained nearest distribution',proportional_budgets(pools=[[(0,1)],[(0,1),(2,3)]],quotas=[1,3],budget=4,mode='strict'),(0,4))
    checks.raises('impossible global budget',ValueError,proportional_budgets,pools=[[(0,1)],[(0,1)]],quotas=[.5,.5],budget=1,mode='strict')
    techniques=['random','terms','entities','whole_entities','domain','keyphrase','ner_domain','ner_keyphrase','triple']
    result=inspect_corpora(corpora=[str(a),str(b)],output_dir=str(root/'preview'),languages=['es','en'],techniques=techniques,terms=['dolor','heart failure'],probability=.2)
    stats=json.loads(result.statistics.read_text())
    checks.equal('four documents reported',stats['total']['documents'],4)
    for name in techniques:
        directory=result.output_dir/name
        report=json.loads((directory/'summary.json').read_text())
        checks.equal(f'{name} global budget',report['selected_words'],int(.2*stats['total']['words']))
        for index,original in enumerate(frames):
            output=pd.read_parquet(directory/f'corpus_{index:03d}.parquet')
            checks.equal(f'{name} corpus {index} size',len(output),len(original))
            checks.equal(f'{name} original text retained',output.text.tolist(),original.text.tolist())
            checks.equal(f'{name} counted words {index}',int(output.n_masked_words.sum()),report['allocated'][index])
    annotated=first.copy()
    annotated['entities_json']=json.dumps([{'id':'T1','start':18,'end':23,'label':'SYMPTOM','text':'dolor'}])
    annotated['n_entities']=1
    path=root/'annotated.parquet';annotated.to_parquet(path,index=False)
    read_corpora(paths=[path])
    checks.check('canonical annotations validated',True)
    special=inspect_corpora(corpora=[str(path)],output_dir=str(root/'entity'),language='es',techniques=['whole_entities'],probability=.2)
    output=pd.read_parquet(special.output_dir/'whole_entities/corpus_000.parquet')
    checks.equal('annotations preserved',output.entities_json.tolist(),annotated.entities_json.tolist())
    checks.raises('M8 not simulated',ValueError,inspect_corpora,corpora=[str(a)],output_dir=str(root/'bad'),language='es',techniques=['M8'])
    args=[sys.executable,'scripts/inspect_corpus.py',str(a),str(b),'--languages','es','en','--output-dir',str(root/'cli'),'--techniques','random','terms','--terms','dolor','heart failure','--probability','.2','--show-examples']
    process=subprocess.run(args,capture_output=True,text=True)
    checks.equal('script calls CLI successfully',process.returncode,0)
    if process.returncode: print(process.stderr)
    checks.check('examples and counts printed','masked_words=' in process.stdout)
    checks.check('saved CLI invocation',(root/'cli/masking.yaml').is_file())
    checks.raises('restricted terms cannot fabricate budget',ValueError,inspect_corpora,
                  corpora=[str(a)],output_dir=str(root/'restricted'),language='es',
                  techniques=['terms'],terms=['dolor'],probability=1,backfill=False)
    under=inspect_corpora(corpora=[str(a)],output_dir=str(root/'underfill'),language='es',
                         techniques=['terms'],terms=['dolor'],probability=1,
                         backfill=False,budget_mode='underfill')
    under_report=json.loads((under.output_dir/'terms/summary.json').read_text())
    checks.equal('explicit restricted term underfill',under_report['selected_words'],1)
    from verify_cpt import fixture
    _,_,base,training,_,_=fixture(root/'entropy-fixture')
    entropy_path=root/'entropy.parquet';training.to_parquet(entropy_path,index=False)
    actual=inspect_corpora(corpora=[str(entropy_path)],output_dir=str(root/'entropy'),
                          language='en',techniques=['M8'],base_model=str(base),probability=.2)
    checks.check('real local entropy selection',json.loads((actual.output_dir/'entropy/summary.json').read_text())['selected_words']>0)
    print('Artifacts:',root)
    return checks.report()


if __name__=='__main__': run(main)
