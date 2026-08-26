from pathlib import Path
import hashlib,json,subprocess

def fingerprint(path):
    h=hashlib.sha256()
    with open(path,"rb") as f:
        for block in iter(lambda:f.read(1024*1024),b""): h.update(block)
    return h.hexdigest()
def version(name):
    try:
        from importlib.metadata import version as get; return get(name)
    except Exception:return None
def write_run_manifest(run_dir,cfg,tokenizer,stats):
    paths=[]
    def visit(x):
        if isinstance(x,dict):
            for k,v in x.items():
                if k=="path" and isinstance(v,str) and Path(v).is_file(): paths.append(v)
                else: visit(v)
        elif isinstance(x,list):
            for v in x:visit(v)
    visit(cfg)
    try: commit=subprocess.check_output(["git","rev-parse","HEAD"],text=True).strip()
    except Exception:commit=None
    manifest={"base_checkpoint":cfg["model"]["checkpoint"],"git_commit":commit,"config":cfg,"seed":cfg["training"].get("seed",42),
      "tokenizer":getattr(tokenizer,"name_or_path",None),"objective":cfg["objective"],"masking":cfg.get("masking"),
      "optimizer":"AdamW","scheduler":cfg["training"].get("scheduler"),"learning_rate":cfg["training"]["learning_rate"],
      "effective_batch":cfg["training"]["batch_size"]*cfg["training"].get("gradient_accumulation_steps",1),
      "precision":cfg["training"].get("mixed_precision","no"),"dataset_statistics":stats.to_dict(),
      "fingerprints":{p:fingerprint(p) for p in sorted(set(paths))},"versions":{x:version(x) for x in ["torch","transformers","accelerate","minicons","faiss-cpu","pyarrow"]}}
    try:
        import torch; manifest["gpu"]=torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
    except Exception:manifest["gpu"]=None
    (Path(run_dir)/"run_manifest.json").write_text(json.dumps(manifest,indent=2,sort_keys=True))
