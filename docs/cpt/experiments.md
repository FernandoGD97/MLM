# Controlled CLM → MLM experiments

Run priorities are P0 M0/M1/M3, P1 M2/M4, P2 M5/M6/M7, P3 optional M8 and later MT studies.
No grid runs by default. Original, CLM-only and MLM-only controls have separate YAMLs from
the generated CLM→policy configs. The smoke generator writes fully executable absolute
paths into a new directory, not cluster-specific paths or hidden service calls.

For a scientific run, replace synthetic model/corpora with explicitly prepared local
artifacts. Start every CLM→MLM branch at the same `base_model` directory and, preferably,
the explicit `parent_sha256` returned by `checkpoint_sha256`. Keep corpus, example order,
effective batch, optimizer/scheduler horizon, masking probability, replacement fractions,
evaluation masks and seeds fixed. Set independent training/downstream seeds deliberately.

```yaml
task: cpt.pretrain_masked
base_model: assets/common-clm/model
documents: assets/train.parquet
validation: assets/dev.parquet
entity_predictions: assets/predicted-train-spans.parquet
output_dir: runs/ner-domain-seed42
random_state: 42
max_length: 128
mask_policy:
  policy: M5
  probability: 0.15
  fusion: score
  weights: [1.0, 1.0, 1.0]
  phrase_projection: maximum
  budget_mode: strict
  mask_seed: 43
training_parameters:
  epochs: 5
  batch_size: 8
  gradient_accumulation: 4
  learning_rate: 0.00002
  scheduler: cosine
  warmup_ratio: 0.05
  data_seed: 42
  evaluation_seed: 44
  evaluation_probability: 0.15
  evaluate_pll: false
```

This scientific YAML requires caller-supplied assets; it is not a smoke fixture and is not
executed automatically. Weights above are examples, never an asserted optimum.

```python
from lab.cpt import compare_pretraining_runs

table = compare_pretraining_runs(
    runs=["runs/random-seed42", "runs/ner-domain-seed42"],
    output_dir="reports/controlled-comparison",
)
```

Comparison rejects changed control parameters, incomplete runs and budget deviations. It
reports parent IDs, seeds, steps, actual prediction-target counts, costs and the selected
dev metric. Set `require_comparable=False` only for explicitly exploratory tables, such as
different seeds or native/MT volume studies. Do not interpret these as controlled-policy
comparisons. Candidate-set, pooling/index and downstream protocol configuration belongs to
the external probe and must be held fixed by the experiment script; the callable's name
is recorded, but its closure cannot be reconstructed automatically.

Checkpoint step 0, each evaluated epoch, final and best are retained. Final test is a
separate explicit evaluation after selection. Fine-tuned downstream metrics must be
reported separately from the frozen representation probe.

Masking schedules and optional generative/embedding KPE adapters are not implemented in
`lab.cpt`. Existing legacy schedule components remain in `encoder_pretraining`; they do not
silently activate in these controlled comparisons.
