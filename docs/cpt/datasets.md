# Corpus, provenance and alignment

Supply Parquet, a DataFrame, or an HF Dataset via its `to_pandas` interface. Loading remote
datasets is an explicit external operation. `doc_id` and `text` must be unique/nonempty
string identifiers and nonempty raw text. Duplicate IDs, bad offsets and invalid types fail.

Existing `source`, `language`, `translation_status`, `original_doc_id` and additional columns
are preserved. Absent metadata are never invented. `entities_json` and `n_entities` become
empty canonical annotation columns, so gold annotations cannot leak into importance.
Predicted NER data arrive separately as `filename,label,start_span,end_span,text[,score,code]`.

Words use canonical UAX #29 segmentation. Character spans are `[start,end)` in original
document coordinates. Each immutable window retains word character extents, model token
positions and per-token offsets. Fast tokenizers are required. Words never split across
windows; tokens straddling multiple UAX words and words larger than a window fail explicitly.
Window retokenization is checked against the full-document token IDs: a boundary-sensitive
tokenizer that changes them is rejected rather than silently changing the experiment.
Whitespace-only tokens are ineligible. Chunking never concatenates documents.

`grouped_split` joins original/translation IDs transitively before assigning train/dev.
The training entrypoints also refuse overlapping IDs, direct original IDs and identical
texts across train/dev. Run `grouped_split` over the combined source tables to protect
longer family chains. Deduplication and language metadata are the caller's explicit input
choices. Do not compare Native vs MT and masking policy simultaneously in the first study.

The current in-memory implementation does not stream massive corpora. `match_corpus_volume` in `lab.cpt.sampling` explicitly matches eligible-subword volume
using whole-document subset feasibility and a caller-supplied budget. `describe_corpus`
reports metadata composition and duplication counts; manifests include these aggregates.
Terminology-specific distributions require the caller's concept inventory and remain an
external experimental report.
