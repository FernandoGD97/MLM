# Phase 0 audit and reuse map

Inspected before implementing CPT:

- Working checkout: `FernandoGD97/MLM`, initially clean, commit
  `447638a9bd1d7862d85a09ff71a2d3bb304c91c3`.
- Architecture reference: `nlp4bia-bsc/ner_lab`, commit
  `8640d4797b081a5f6b66d9b45c11269542a79a26`.
- Operational reference: `nlp4bia-bsc/TransCPT`, commit
  `e9dd333902e4fedaa888d7167431392bf7a96fd9`.

The user explicitly chose implementation **in MLM**, adapting `lab` there. External
reference checkouts were not modified. The imported `lab.core`, `lab.ner`, `lab.nel`, CLI,
docs and verification baseline preserve the upstream design. Integration adds only `cpt`
to the string-only core task namespace list and the layering verification's task list.
Original upstream decisions remain intact; CPT decisions append to their register.

## Reuse

| Existing component | Use |
|---|---|
| `core.segmentation.split_into_words` | canonical UAX #29 word occurrences |
| canonical corpus/span column names | raw corpus plus supplied metadata; predicted spans separately |
| `core.tasks` and `lab.cli` | Python/YAML parity, lazy task resolution, one CLI |
| `core.scoring.score_spans` | exact labelled entity metrics in external examples |
| `nel.evaluation.retrieval_metrics_from_codes` | concept-deduplicated recall/MRR in external probes |
| upstream NER/NEL code | retained for independent downstream fine-tuning/linking |
| `verification/_harness.py` | plain synthetic behavioral scripts without mandatory pytest |

The CPT implementation owns its corpus preparation because canonical `validate_corpus`
returns only canonical columns and would discard source/translation metadata. It does not
change that core behavior. High-level training uses frozen typed parameter records; the
YAML mappings have those exact schemas and reject unknown keys.

## Verified TransCPT findings

References below refer to the inspected `trans_cpt/training.py` and `utils.py`.

- `training.py` declares module-global accelerator/logger and assigns them in
  `training_pipeline`. CPT instead owns an Accelerator per invocation.
- `concat_chunk_dataset` concatenates token lists across documents and drops incomplete
  tails. CPT preserves document boundaries, offsets and all eligible tokens.
- Split size computes `train_size=N*(1-r)` then `test_size=r*train_size`; the resulting
  requested sizes need not exhaust N. CPT requires explicit disjoint corpora and exposes
  transitive original/translation grouping before any tokenization.
- Evaluation masks are precomputed, but the masking function does not use its `seed`
  argument directly and printed examples expose token IDs. CPT masks derive from stable
  occurrence IDs and separate seeds; logs contain aggregate numeric statistics.
- Scheduler construction is present, but the training loop neither accepts nor calls the
  constructed scheduler. CPT advances it once per successful optimizer update.
- Training steps divide by process count before the prepared-loader length is available;
  the training loop does not implement gradient accumulation. CPT computes update groups
  explicitly and normalizes loss over all actual targets in each group.
- Train/eval average batch mean losses rather than total target-weighted NLL. Both are
  exponentiated and labelled perplexity for MLM. CPT distinguishes fixed-mask CE, word
  PPPL, entity PPPL and strictly causal PPL.
- Saving keeps model/tokenizer weights, not optimizer/scheduler/RNG trajectory state.
  CPT saves full Accelerate state and explicit epoch-boundary continuation state.
- `training_pipeline` prints the input mapping and a runtime library-path variable. CPT
  logs no runtime environment dump or raw clinical samples.
- Existing multi-process behavior was inspected, **not executed**. CPT rejects distributed
  launches rather than claiming those operational assumptions were verified.
- The preprocessing adapter downloads a dataset through environment-specific repository
  configuration. CPT consumes explicit local inputs; downloads are separate user actions.

No external source files, credentials, environment files or cluster paths were copied from
TransCPT. These are intentional scientific/operational differences, not compatibility fixes.

## Legacy MLM corrections

`encoder_pretraining.objectives` now delegates to the verified attention converter: cached
module flags change with configuration, fixing future-token leakage. The older tokenizer
path now builds special-token masks without the unsupported fast-tokenizer operation,
including a checked modern-backend path. All fourteen existing tests pass.
