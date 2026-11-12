# Intrinsic and downstream evaluation

Fixed-mask validation always uses random subword masking with its own seed and probability,
independent of the training policy. Predictions are scored with summed CE divided by the
actual count of valid targets, plus exact masked-token accuracy. Zero-target evaluation
raises. MLM CE exponentiation is never reported as autoregressive perplexity.

CLM uses the same masked-LM head with explicitly shifted next-token labels and genuinely
causal BERT/RoBERTa attention. Padded and special targets are excluded. Causal PPL is exp of
target-weighted causal CE, and evaluation rejects a model in the wrong attention mode.

`pseudo_likelihood` supports two explicit bounded-window conditionings:

- `within_word_l2r`: target and later subwords of its UAX word are masked; earlier pieces
  and other words remain visible. Every eligible token in every document window is scored.
- `whole_entity`: every token of a complete entity is masked together. Each occurrence is
  scored separately, including repeated/overlapping entities. Truncated occurrences are
  skipped and counted. This is a pseudo-likelihood, not a joint entity likelihood.

Both PPPLs normalize by scored subwords. No suffix of a long document is silently dropped;
context is explicitly bounded by its nonoverlapping training-compatible windows. Evaluation
temporarily restores bidirectional attention and restores original mode afterward.

Run `cpt.evaluate_checkpoint` on the original encoder, CLM checkpoint, each branch's saved
step-0/best/final checkpoint. Training evaluates and saves step 0 and each configured epoch.
Word PLL can run each evaluation epoch with `evaluate_pll: true`; entity PLL is a composable
Python function requiring explicit predicted occurrences.

## Linking first, then NER

`examples/cpt_downstream.py` composes frozen representation retrieval with the existing
`lab.nel` metrics and the canonical exact NER scorer. Alias vectors rebuild on every probe;
there is no cross-checkpoint index cache. Aliases collapse to concept IDs at their maximum
cosine score. Both mention-only and contextual mention pooling use normalized mean vectors,
excluding padding/special tokens. Contextual mode pools the supplied mention's subwords.

Metrics include Recall@1/5/25, MRR, gold-vs-best-other ranking margin (when defined), and
caller-defined seen/unseen concept and surface subsets. Missing gold concepts affect recall;
margin is only defined when gold and a competitor exist. NER uses labelled exact span P/R/F1
through the upstream scorer, never token accuracy.

Pass a callable `evaluation_probe(model,tokenizer,windows,epoch)` to `pretrain_masked` for
per-epoch dev monitoring. It is an external composition point; CPT imports no NER/NEL code.
For a fixed valid EL protocol, choose `selection_metric: recall@1` and
`greater_is_better: true` in Python and return that metric from the probe. Without a probe,
the explicit default is fixed-mask dev NLL, never training loss. Arbitrary Python probes
are intentionally not serialized into YAML.

Final-test data have no argument in training, so they must be evaluated separately after
checkpoint selection. Fine-tuning downstream uses the unchanged `lab.ner`/`lab.nel`
facilities externally; it is not the frozen probe and has not been verified on real models.
