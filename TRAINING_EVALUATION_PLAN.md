# Training and evaluation plan

1. Add one shared, target-weighted training engine and thin MLM/CLM entrypoints.
2. Evaluate immutable epoch 0, then checkpoint and evaluate every configured epoch.
3. Add fixed-mask MLM CE, causal PPL, strided `within_word_l2r` PLL/PPPL, and safe temporary bidirectional mode.
4. Add zero-shot dense EL with per-epoch candidate re-encoding, terminology-level reuse, concept-level metrics, and monitor/final-test roles.
5. Persist tidy metrics, manifests, fingerprints, predictions, and a NER adapter interface; validate with tiny deterministic tests.

Each stage reuses the existing data, masking, entity, and collator modules.
