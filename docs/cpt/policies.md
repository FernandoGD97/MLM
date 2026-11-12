# Importance and target selection

`MaskPolicy` contains one responsibility: prediction-target selection and corruption.
Scores describe occurrences, types, spans or token positions explicitly. Only training
corpora fit lexical statistics; dev/test never do. Confidence is an optional detector
signal, not a claim of clinical importance. Fusion weights are experimental choices.

| ID | Selection units and signal |
|---|---|
| M0 | shuffled subword positions, or whole UAX words with `random_unit: word` |
| M1 | independent whole words prioritized by predicted entity spans |
| M2 | whole entities; overlapping/nested entities merge into indivisible components |
| M3 | whole words ranked by TF-IDF, TF-ICF, smoothed log odds or frequency ratio |
| M4 | whole words with explicitly projected local lexical keyphrase scores |
| M5 | NER + domain words |
| M6 | NER + keyphrase words |
| M7 | NER + domain + keyphrase words |
| M8 | whole-word masked predictive entropy; requires `adaptive_enabled: true` |

M1/M2/M5/M6/M7 consume a separate canonical span table; corpus gold-looking fields are
discarded. `score_entities` supports binary/confidence, type weights and optional inverse
square-root concept frequency. The public component accepts reliable concept-frequency
maps; the high-level YAML currently exposes binary/confidence and type weights only.
No detector runs inside the collator.

The concrete KPE extractor enumerates exact contiguous alphanumeric n-grams and ranks their
mean training-fitted word relevance. It is inexpensive and reproducible, not an embedding
model. There are no nonfunctional MDERank/PromptRank/Attention-Seeker adapters.

Phrase projection is always explicit:

- `maximum`, `mean`, `sum`: aggregate covering phrase scores at each word occurrence.
- `uniform`: add phrase score divided by its number of words.
- `length_normalized`: distribute by character length, excluding intervening whitespace.

Repeated spans remain separate occurrences. Missing word evidence is zero; missing/nonfinite
scores and misaligned phrase spans fail. High-level preparation discards and counts
truncated/misaligned entity occurrences rather than projecting a partial entity.

Fusions include score fusion after percentile normalization, weighted reciprocal ranks,
RRF (constant 60), hierarchical priority and disjoint budgets. Zero evidence stays zero.
Hierarchical order is NER, domain, KPE. Disjoint overlap ownership goes to the first active
positive-weight detector. Integer quotas use weighted largest remainder; unfillable quotas
are backfilled without splitting words. Global feasibility can require reallocating quotas;
requested and actual detector allocations are logged separately. No quota equality is
claimed after reallocation. Detector contribution counts may overlap under score fusion.

## Shared budget

`N` is the number of eligible nonspecial, nonpadding subword positions with aligned words.
`B = floor(probability * N)`. A suffix subset-sum feasibility table guides ranked selection.
Strict mode reaches B when compatible units exist, otherwise raises. `budget_mode: underfill`
is the explicit alternative; deviation is recorded and controlled run comparison refuses
it. Oversized whole entities are skipped, never partially selected. M2 backfills only from
nonentity subwords; word policies backfill whole words. Baseline M0 has no priority backfill.

Replacement is one sampled action per selected unit: 80% `[MASK]`, 10% random nonspecial
vocabulary entries, 10% unchanged, configurable. Random draws may reproduce the original
ID, so random-replacement actions and actual corruption counts differ. All selected
positions carry original labels; every unselected or padded position carries `-100`.

Epoch logs aggregate eligible subwords, selected targets, budget/deviation, corruption,
mask/random/unchanged actions, words/entities, entity target share, mean selected unit length,
backfill fraction, skipped spans, unique word types/concepts and detector contributions.
Unique counts use epoch-level sets rather than sums of batch uniques. Distributed launches
are rejected until aggregation and execution are verified.
