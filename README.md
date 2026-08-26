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

## Per-epoch training and representation probes

The shared `TrainingEngine` powers two thin objective-specific entrypoints:

```bash
python scripts/train_clm.py --config configs/train_clm.yaml
python scripts/train_mlm.py --config configs/train_mlm.yaml
```

Both evaluate the untouched base checkpoint as epoch 0, then retain every epoch checkpoint.
MLM training and validation aggregate negative log likelihood by predicted mask count and
never label `exp(MLM CE)` as perplexity. CLM reports true target-weighted causal PPL. During
CLM evaluation, PLL/PPPL and dense EL run inside an exception-safe temporary bidirectional
context and causal mode is restored afterward.

The strided PLL backend scores every token of long documents without silent suffix
truncation and defaults to Kauf and Ivanova's `within_word_l2r`. Its deterministic document
selection is persisted in `pppl_eval_manifest.json`. Dense EL is a zero-shot representation
probe: aliases and mentions are encoded by the same epoch checkpoint, normalized inner
product implements cosine retrieval, aliases are deduplicated at concept-ID level, and one
terminology index is shared by all configured datasets within that epoch. Candidate
embeddings are deliberately rebuilt at the next epoch.

EL datasets marked `monitor` run at epoch 0 and each evaluation epoch. A `final_test` is
only read at the final epoch and must never be used for checkpoint selection, early stopping,
or hyperparameter tuning—even if an input filename happens to contain the word “test”.
Outputs include a tidy `epoch_metrics.parquet`, `epoch_summary.json`, per-epoch JSON metrics,
optional per-mention Parquet predictions, the resolved config, and a version/fingerprint-rich
run manifest. `NerLabEvaluator` is intentionally only an interface: a pure CPT checkpoint has
no trained NER classification head and is not presented as if it did.

Implementation structure was reviewed against the public designs of
[TransCPT](https://github.com/nlp4bia-bsc/TransCPT),
[lm-pseudoperplexity-evaluation](https://github.com/DataTools4Heart/lm-pseudoperplexity-evaluation),
[ner_lab](https://github.com/nlp4bia-bsc/ner_lab), and
[CardioLM](https://github.com/DataTools4Heart/CardioLM). This implementation does not copy
their loops: it uses target-weighted NLL aggregation, distinguishes causal PPL from masked
PLL, isolates expensive external evaluation to the main process, and persists the exact
evaluation subset and scientific dataset role.

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
