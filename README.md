# Modular continual pretraining for encoders

Research-oriented data preparation and objective plumbing for **dynamic MLM**, temporary
**CLM on the same encoder backbone**, **CLM → MLM**, and raw-text **entity-first MLM**.
Gold-looking Parquet fields `entities_json` and `n_entities` are deliberately discarded;
all entity spans are newly inferred from raw text. Other metadata is retained opaquely so
future data-mix experiments remain independent from objective experiments.

## Design and scientific controls

The flow is `Parquet → document-bounded tokenization → EntityProvider → schedule + masking
policy → replacement → collator → Trainer`. Preprocessing never masks. Schedules decide
**how much**, policies decide **what**, and replacement decides **how**, allowing isolated
ablations. Entity selection uses whole spans and a fixed total prediction budget; the
entity budget fraction is an **EXPERIMENTAL DESIGN CHOICE**, not a claimed optimum.

`EntityProvider` includes none, external gazetteer, and arbitrary Hugging Face token-
classification checkpoints. Detection happens on raw text and fast-tokenizer offsets
project spans to subwords. Content remainder filtering never removes text, retains numbers,
doses and units, and has no default stopword list (so negations such as *no/sin/not/without*
remain eligible).

## CLM safety

CLM uses the model's existing masked-LM head and the **same weights**, but changes BERT or
RoBERTa's native attention path to causal and makes position `i` predict token `i+1`.
Transitions checkpoint weights and restore bidirectional attention for MLM. Unsupported
architectures fail fast. In particular, ModernBERT is intentionally not advertised as
supported until its installed Transformers implementation exposes a verified, leakage-free
same-backbone causal path; silently substituting a decoder is forbidden.

The phase configs use explicit step or token budgets. Token budgets are converted to steps
from batch size and observed mean sequence tokens, avoiding epoch-dependent comparisons.
Distributed world size and gradient accumulation should be included in a future exact global
token accountant; this current conversion is an **EXPERIMENTAL DESIGN CHOICE** for the
single-process baseline.

## Usage

```bash
pip install -e '.[test]'
pytest -q
python scripts/inspect_masking.py --config configs/mlm_entity_first.yaml --examples 20
python train.py --config configs/clm_to_entity_mlm.yaml
python train.py --config configs/mlm_random_15.yaml
```

The inspection command prints original/masked text and `ENTITY`, `CONTENT`, or `RANDOM`
origins before training. Runtime configurations and seeds should be archived with output.

## Literature provenance

The separation and controlled compute comparison of CLM, MLM, and biphasic training follows
the experimental questions in [Gisserot-Boukhlef et al.](https://arxiv.org/abs/2507.00994), [Touchent & de la
Clergerie](https://arxiv.org/abs/2605.12438), and [AntLM](https://aclanthology.org/2024.conll-babylm.29/) (CoNLL 2024). Whole-entity priority plus generic
remainder is inspired by [EntityBERT](https://aclanthology.org/2021.bionlp-1.21/) (BioNLP 2021), [domain whole-concept masking](https://aclanthology.org/2025.repl4nlp-1.6/) (RepL4NLP
2025), and [mask-specific biomedical losses](https://aclanthology.org/2024.naacl-long.280/) (NAACL 2024). [Difference-Masking](https://aclanthology.org/2023.findings-emnlp.881/) (EMNLP 2023)
motivates a future replaceable informative-remainder policy; it is not implemented here.

No downstream task, NER/NEL training, ontology objective, catastrophic-forgetting method,
native/translation experiment, tokenizer training, contrastive learning, LoRA, or bundled
biomedical NER checkpoint is included.
