"""Configurable terminology/query ingestion; no benchmark-specific columns."""
from dataclasses import dataclass

@dataclass(frozen=True)
class Mention:
    doc_id:str; mention:str; gold_concept_id:str; text:str|None=None; start:int|None=None; end:int|None=None; strata:dict|None=None

def read_rows(path):
    import pyarrow.csv as csv, pyarrow.parquet as pq
    return (pq.read_table(path) if str(path).endswith((".parquet",".pq")) else csv.read_csv(path)).to_pylist()
def load_terminology(config):
    rows=read_rows(config["path"]); return [(str(r[config["concept_id_column"]]),str(r[config["term_column"]])) for r in rows]
def load_mentions(config):
    c=config["columns"]; result=[]
    for r in read_rows(config["path"]):
        text=r.get(c.get("text")) if c.get("text") else None; start=int(r[c["start"]]) if c.get("start") and r.get(c["start"]) is not None else None
        end=int(r[c["end"]]) if c.get("end") and r.get(c["end"]) is not None else None
        mention=r.get(c.get("mention")) if c.get("mention") else None
        if mention is None and text is not None and start is not None: mention=text[start:end]
        if text is not None and start is not None and mention is not None and text[start:end]!=mention: raise ValueError("Configured mention disagrees with text[start:end]")
        strata={name:r[col] for name,col in config.get("strata_columns",{}).items()}
        result.append(Mention(str(r.get(c.get("doc_id"),"")),str(mention),str(r[c["gold_concept_id"]]),text,start,end,strata))
    return result
