# Biomedical continual pretraining

Implemented in **FernandoGD97/MLM**, following the user's explicit repository choice.
The `lab` namespace, canonical corpus/span tables, lazy task dispatch, CLI and the original
NER/NEL implementations come from `ner_lab` at the revision in `UPSTREAM_NER_LAB.md`.
The older `encoder_pretraining` package remains available. CPT never imports NER or NEL;
composition lives in `examples/cpt_downstream.py`.

## Install and run

Python 3.11+, tested on Python 3.12, PyTorch **2.10.0 CPU**, Transformers **5.22.2**.

```bash
python -m venv .venv
. .venv/bin/activate
pip install torch==2.10.0 --index-url https://download.pytorch.org/whl/cpu
pip install -e '.[cpt]'
python examples/cpt_smoke.py --output-dir outputs/cpt-smoke
lab run outputs/cpt-smoke/original.yaml
lab run outputs/cpt-smoke/clm.yaml
lab run outputs/cpt-smoke/M0.yaml
lab run outputs/cpt-smoke/M1.yaml
lab run outputs/cpt-smoke/M3.yaml
```

Asset generation never trains. These YAMLs execute local synthetic experiments; their
numbers are software checks, **not evidence of biomedical improvement**. Every M0–M8 YAML
uses the identical CLM checkpoint. `mlm-only.yaml` instead starts at the original encoder.
Generate into a new directory; neither the generator nor training overwrites runs.
Scientific configs are described in [experiments.md](experiments.md).

```bash
OMP_NUM_THREADS=2 python verification/verify_cpt.py
python verification/run_all.py
python -m pytest -q  # optional pytest: legacy MLM checks only
```

`verify_cpt.py` needs no network or real corpora. It retains temporary checkpoints and prints
the artifact directory. `CPT_VERIFY_OUTPUT` selects a fresh explicit directory instead.
Inherited checks additionally expect cached BERT/GPT-2 tokenizers and optional external
equivalence repositories. See [verification status](STATUS.md) for actual results.

## Public functions

Import from `lab.cpt`; arguments are named, results are frozen records where appropriate.

| Function | Contract |
|---|---|
| `prepare_corpus` | DataFrame/Dataset/Parquet → canonical corpus plus supplied metadata; optional explicit output |
| `tokenize_corpus` | corpus + fast tokenizer → document-bounded windows with UAX words, offsets and subword maps |
| `score_domain_words` | explicit training/reference texts → immutable fitted lexical scores/provenance |
| `score_entities` | one document's canonical predicted spans → typed occurrence importance |
| `extract_keyphrases` | exact local n-grams ranked using fitted lexical relevance |
| `project_phrase_scores` | explicit span-to-word projection; misaligned spans raise |
| `fuse_scores` | weighted percentile score, weighted rank or reciprocal rank fusion |
| `select_mask_positions` | window + `MaskPolicy` + `WindowSignals` → targets, units, budget and counters |
| `pretrain_causal` | local original encoder + train/dev corpus → causal checkpoint artifacts |
| `pretrain_masked` | local shared checkpoint + explicit policy/signals → one comparable branch |
| `evaluate_pretraining` | in-memory model + windows → fixed-mask target-weighted CE/accuracy |
| `evaluate_checkpoint` | saved checkpoint + corpus → metrics and content hashes; optional output |
| `compare_pretraining_runs` | completed branch artifacts → controlled comparison table |

Additional composable APIs live in their responsible modules: `grouped_split`,
`pseudo_likelihood`, `predictive_entropy`, `checkpoint_sha256`, `set_attention_mode`,
`verify_attention`, `TrainingParameters`, `MaskPolicy`, `WindowSignals` and `train`.
Task tables register `cpt.prepare_corpus`, `cpt.pretrain_causal`, `cpt.pretrain_masked`,
`cpt.evaluate_checkpoint`, `cpt.compare_pretraining_runs` for the existing `lab run` CLI.

## Guides

- [Policies, budgets and phrase projection](policies.md)
- [Datasets and alignment](datasets.md)
- [Evaluation](evaluation.md)
- [Reproducibility and resume](reproducibility.md)
- [Scientific experiments](experiments.md)
- [Audit and architecture decisions](AUDIT.md)
- [Verification status and limitations](STATUS.md)

## Exploración de Parquet

[Script único de estadísticas, segmentación multilingüe y masking proporcional](inspection.md).
