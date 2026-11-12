# Implementation and verification status

The request's complete acceptance criteria are **not all verified**. CPT's local synthetic
workflow is implemented; the inherited full regression run is blocked by unavailable
tokenizer downloads, and hardware/scientific claims require additional evidence.

## Verified so far

- CPU suite: **186 checks passed, 0 failed** on the final source.
  Artifacts: `outputs/cpt-final`; log: `outputs/cpt-final/verification.log`.
- Legacy MLM suite: 14 passed.
- Ruff check/format: CPT source and its examples pass.
- CLM→common checkpoint→M0–M8 training, including epoch-0/1/2 validation and full checkpoints.
- BERT and RoBERTa causal non-leakage, bidirectional contextual dependency, backward and
  exact saved/reloaded parameters.
- Exact budgets and impossible subset handling, whole words/entities, nested spans,
  phrase projection, domain scores, fusion variants, dynamic masks and replacement labels.
- Fixed-mask evaluation, exact uniform-logit CE/PPPL, entity PLL, concept-deduplicated EL
  and canonical exact NER scoring on synthetic data.
- Bitwise-equal CPU uninterrupted/resumed trajectories for CLM, M0 and cached M8; cache
  refresh frequency and resumed best-checkpoint location are checked.
- Public Python APIs, task resolution and execution through `lab run`.
- Manifest/log artifacts contain aggregates and content hashes, not raw corpus/lexical text.

## Inherited regression run

`python verification/run_all.py` was executed. With CPT included in the runner (185 checks at that revision, before the last
legacy-tokenizer regression check):

- Layering: 156 passed.
- Core data: 192 passed, 2 skipped (optional real corpora).
- NEL: 62 passed, 2 skipped (external equivalence checkout and missing tokenizer cache).
- Nine other scripts failed while resolving uncached `bert-base-uncased`/`gpt2` tokenizers.
  A direct HTTPS probe returned proxy CONNECT **403** for `huggingface.co`.

The `huggingface.co` domain addition was saved to the environment draft; saving does not
apply it. Apply the network change in environment settings, cache the required tokenizers,
then rerun the full suite. Redirect domains, if any are actually requested, may also need
explicit access. No TLS, checksum, assertion or verification bypass was used.

## Remaining limits

- BERT/RoBERTa only, eager attention; other architectures fail before training.
- Single-process CPU verified. Single GPU/mixed precision and TensorBoard are implemented
  but untested here; distributed execution is explicitly rejected.
- Resume is at epoch boundaries only. Framework state files must be trusted.
- No real biomedical checkpoint/corpus experiment or downstream fine-tuning was run.
- No conclusion about masking-policy quality follows from the synthetic results.
- Frozen EL probes support explicit Python composition; complete scientific probe closure
  configuration and final-test isolation remain the experiment script's responsibility.
- In-memory corpus/statistics/tokenization; no massive streaming corpus guarantee.
- Exact word boundaries are required; unsupported tokenizer boundary behavior raises.
- Offline lexical KPE only; no claim of implementing MDERank/PromptRank/Attention-Seeker.
- Energy/emissions and new CPT masking schedules are absent, documented optional extensions.

## Phase continuation record

Phases 0–5 have synthetic implementations and checks. Phase 6 has intrinsic and external
frozen downstream synthetic checks; real-model downstream validation is unrun. Phase 7
has manifests, resume, CLI/YAML generation and comparison; full inherited regression remains
blocked as described above. Next external action: apply network access, then run the complete
inherited suite and diagnose any independent compatibility failures before acceptance.
