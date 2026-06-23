# HACA-Sent v3 — Training Dataset Datasheet

**Status:** in construction (Stages 1–4 of the domain-adaptation plan).
**Purpose:** in-domain training data to close the broadcast domain gap (target macro-F1
≥ 0.70 on the frozen `domaine_reel_v2` gold test).
**Labeling rule:** [ANNOTATION_RUBRIC_V3.md](ANNOTATION_RUBRIC_V3.md) (content-valence).

> This is a **disclosure datasheet**. It records exactly how the data was made so the
> training set is defensible in the internship report. Counts are filled as stages complete.

---

## 1. Composition (filled per stage)

| Source | Rows | neg | neu | pos | label_source | notes |
|---|---|---|---|---|---|---|
| Real SRT (clean, re-segmented) | 1199 | 191 | 973 | 35 | `claude-manual` | from 12 files in `data/raw/srt/` |
| Synthetic (Claude-authored) | 189 | 40 | 20 | 129 | `claude-synth` | `synthetic=true`, pos-weighted |
| **Total `haca_train_v3.csv`** | **1388** | **231** | **993** | **164** | | 14% synthetic |

The real pool is genuinely neu-heavy and pos-scarce (3% pos) — sincere positive valence is rare
in news/explainer/debate broadcast. Synthetic generation lifts `pos` 35 → 164 (×4.7) so the
model can learn the positive content-valence concept; training additionally uses class-weighted
**focal loss** + ×3 oversampling of in-domain `pos` (see `src/finetune.py` `marbertv2-haca`).

**Stage 1 extraction (`src/build_haca_pool.py` → `haca_pool_v3.csv`):** 1471 candidate
utterances → **1199 clean** (label-ready) + 272 garbled (quarantined). A further **372
candidates were dropped for test-leakage** (shared a word-5-gram with `domaine_reel_v2` /
`domaine_reel`), since the gold test is drawn from the same source files. Avg clean utterance
≈ 175 chars. Garble filter: `src/asr_quality.py` (function-word density + length + script +
repetition).

## 2. Provenance

- **Real text:** 12 Moroccan-Darija broadcast files (`data/raw/srt/*.srt`) — TV debate/talk
  shows + long-form YouTube explainers (tax, procurement, markets, health, history,
  anti-corruption). Extracted and re-segmented by `src/extract_utterances.py` +
  `src/asr_quality.py`. No new files were added (corpus-limited by design).
- **Real labels:** authored by Claude, one utterance at a time, under Rubric v3. Every `pos`
  and a sample of borderline `neu/neg` was double-checked. No external LLM API was used.
- **Synthetic text + labels:** authored by Claude to fit Rubric v3 and the real broadcast
  register, deliberately over-representing `pos` (structurally rare in news content). No
  external LLM API was used. Flagged `synthetic=true`.

## 3. Integrity rules

- The gold test `domaine_reel_v2.csv` (194, human-annotated) is **never** relabeled or used
  for training. `haca_train_v3.csv` is deduplicated against its text.
- Synthetic rows are deduplicated against the real pool and the test set.

## 4. Known limitations

- Training labels are model-authored (silver), not human gold — disclosed here.
- Synthetic positives may under-represent the messiness of real ASR; mitigated by keeping a
  real-positive core and reporting the synthetic share.
- Corpus topic skew (economics/politics/governance heavy) inherited from the 12 source files.
