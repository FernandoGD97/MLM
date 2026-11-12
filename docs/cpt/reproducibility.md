# Reproducibility and artifacts

Training, data ordering, masking and evaluation seeds are separate. Mask seeds derive from
SHA-256 of `(mask_seed,doc_id,window_index,epoch)`, independent of batch/worker order. There
are no global accelerators/loggers or mutable global RNGs in the corruption path.

Each fresh output directory contains:

```text
run_manifest.json
resolved_config.yaml
run.log
events.jsonl
epoch_metrics.parquet
training_summary.json
checkpoints/epoch_000/{model/,accelerate/,training_state.json}
checkpoints/epoch_001/...
evaluations/epoch_000.json
```

The initial manifest precedes optimization. Atomic final updates record completed/paused/
failed status, exception type (without potentially sensitive text), wall time, code version,
installed repository commit/dirty state, parent checkpoint hash, corpus fingerprints,
detector input fingerprint and parameters, policy/projection/fusion/budget/replacement/seeds,
architecture, software/hardware and the optimization plan. Raw clinical text and lexical
tokens are not logged. No text-sample logging is enabled.

The common checkpoint hash includes model/tokenizer file names and contents. Every branch
verifies that the parent remains unchanged. Checkpoints are written only to new directories.
TensorBoard is optional (`training_parameters.tensorboard`) and requires the separately
installed `tensorboard` package. No external tracker is required. Energy/emissions are not
implemented; GPU-hours and memory are recorded when a GPU exists.

## Resume semantics

`resume_from` names an `epoch_NNN` directory, not its `model` child. Restore Accelerate's
model, optimizer, scheduler, mixed-precision state and Python/NumPy/Torch RNG state, plus
epoch/global step, processed tokens, metrics, best-checkpoint location, masking epoch and
the adaptive entropy cache. Data order is reconstructed per epoch. Mid-epoch resume is
not offered. Only load checkpoints you produced/trust: framework RNG state uses pickle.

The full plan, including corpus hashes and original parent checkpoint, must match. Resume
into a **new output directory**, preserving the original run and any earlier best checkpoint.
For interruption tests, set `epochs: 2` from the start and `stop_after_epoch: 1`; do not
change the scheduler horizon by changing epochs when resuming.

CPU continuation is checked against uninterrupted training for CLM, M0 and cached M8.
GPU/distributed bitwise determinism is not claimed. Distributed execution fails early;
single-GPU/mixed precision paths exist through Accelerate but were not exercised here.
