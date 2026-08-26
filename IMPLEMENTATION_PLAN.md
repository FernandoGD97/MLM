# Implementation plan

1. Load Parquet, retain metadata, tokenize without masking, chunk per document, and report corpus statistics.
2. Add deterministic dynamic random MLM and replacement policies.
3. Add same-backbone causal mode, shifted CLM targets, and leakage tests.
4. Add token/step-budgeted phase transitions and checkpoints.
5. Add raw-text entity providers and offset projection.
6. Add whole-entity-first masking, schedules, weighted loss metadata, diagnostics, and inspection CLI.

Each phase is independently configured and tested before it is composed with later phases.
