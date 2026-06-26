# Challenges & Difficulties — HACA Sentiment/Tonality Benchmark

A consolidated account of the hard problems hit while building the multilingual
sentiment benchmark and the deployable tonality pipeline. Grounded in
[`FINDINGS.md`](FINDINGS.md), [`FINETUNING.md`](FINETUNING.md),
[`DEPLOYMENT.md`](DEPLOYMENT.md) and [`PIPELINE.md`](PIPELINE.md); this file
collects the obstacles in one place rather than the results.

---

## 1. Modelling & accuracy challenges

### 1.1 The domain gap (the central finding)
Models that score well on public datasets collapse on real HACA broadcast content.
Macro-F1 drops by **−0.30 to −0.41** going from public Darija tweets to the
194-utterance real-domain set (e.g. MARBERTv2: **0.844 → 0.441**). The gap is
**structural, not accidental** — it affects every model, including the best ones.
Causes: a measured/journalistic register vs. an emotional tweet register, a
75%-neutral broadcast distribution vs. a positive-majority tweet distribution (so
MAC-trained decision boundaries are biased toward pos/neg), and broadcast genres
(religious shows, parliamentary debates, economic documentaries) absent from training.

### 1.2 Per-utterance accuracy ceiling (~0.46–0.52)
On broadcast content, **all** models plateau at ~0.45–0.52 macro-F1 at the utterance
level — calibrated fine-tuned encoders and rubric-prompted LLMs alike — and this is
**invariant to the annotation rubric** (verified on v2 and v3). This drove the core
design decision: don't ship a per-utterance classifier; **aggregate to the
segment/programme level and abstain on noise** (see §3).

### 1.3 The positive class is essentially unmeasurable
Positive sentiment is structurally rare in broadcast news. The real-domain set has only
**10–20 positive examples**, so F1-pos is extremely high-variance (0.08–0.32). Despite a
focal loss, class-weighting, and ×3–4 oversampling of positives, fine-tuned encoders
stayed **blind to the positive class** (F1-pos 0.13–0.19). Removing the MAC tweets
("haca-only") changed nothing — the hypothesis that MAC polluted the positive signal was
**wrong**.

### 1.4 The evaluation set is the real bottleneck — not the model or training data
The decisive constraint is the **size and representativeness of the eval set**:
- Re-labelling under a consistent rubric (v3) did **not** improve the encoder
  (0.452 vs 0.459) — refuting the "it's just annotation misalignment" intuition.
- A larger, balanced set (`domaine_reel_v4_balanced`, 450 ex.) sent everyone to 0.7+
  **but is a worse benchmark**: the jump came entirely from clean, prototypical
  **synthetic** utterances, easy for any model (even a zero-shot LLM that never saw
  them). "Easy ≠ good." The honest number stays ~0.46–0.49.
- A positive class of 16–20 examples at Cohen's κ=0.78 cannot be measured reliably. The
  current corpus is only **12 source files, positive-poor**.

### 1.5 Annotation subjectivity
The "right" label depends on annotation philosophy: strict (does the presenter *express*
sentiment?) vs. content-valence (is what's *described* negative/positive?). The same 194
utterances annotated three ways (strict / Gemini-broad / HACA-v2) **reorder the model
ranking** — xlm-t is 4th on strict but 1st on broadened. v2↔v3 agreement is 88.7%
(κ=0.784), and even that moves the positive class by 20%. The HACA content-valence
rubric (v3) had to be defined explicitly to make the task well-posed.

### 1.6 The scale-inversion surprise
The 9B LLM is **worse** than the 2B on binary arabizi (0.754 vs 0.978) while being better
on 3-class darija (0.833 vs 0.727). Not a bug — a **calibration artifact**: the 9B emits
"neutral" when uncertain (42% of its arabizi predictions), which helps on 3-class but is
pure error on a binary test set with no neutral class. Bigger ≠ better, and test-set
class structure drives apparent quality.

### 1.7 Neutral-vs-negative confusion across the board
Every model struggles on the neutral/negative boundary in broadcast context, with
opposite failure modes: xlm-t and camelbert-da **over-predict negative** (camelbert-da
recall-neg 0.905 but neu recall 0.234), while fine-tuned encoders **over-predict
neutral** (MARBERTv2 recall-neg 0.16). None find the balance.

### 1.8 Encoder vs LLM trade-off has no clean winner
Calibrated encoder (~0.52, ~3 ms) and rubric-prompted LLM (~0.50, ~185–450 ms) tie on
macro-F1 but have **opposite profiles**: the encoder is fast and strong on neu/neg but
blind to positive; the LLM is the only thing that detects positive (recall 0.35–0.70) but
under-predicts neutral, so its edge evaporates on neutral-heavy (i.e. realistic) streams.
The deployment recommendation had to be conditional rather than a single model.

### 1.9 French and arabizi coverage gaps
The real-domain set is 100% Arabic-script Darija — there are **no French and no arabizi
broadcast utterances**, so those models' broadcast performance is untested. French
fine-tunes (xlm-r-haca / camembert-haca) also never reliably beat off-the-shelf
`xlm-sentiment` on the French gold (all within 0.43–0.49 noise), so that fine-tuning
effort did not pay off.

---

## 2. Data-engineering challenges

### 2.1 Garbled ASR fragments
Many SRT cues are incoherent auto-transcription (~36 of 194 real-domain utterances came
from poor SRTs). Left in, they inflate the neutral class and poison predictions.
Solution: a **language-agnostic quality gate** (`asr_quality.py`) scoring function-word
density (Arabic + French), script ratio, length, and repetition — garbled cues are
**excluded from scoring (abstention)**, not mislabelled.

### 2.2 Train/test leakage
The training pool is drawn from the same SRT files as the test set, just with different
segment boundaries — a content-level leakage risk. Solution: a **5-gram overlap filter**
(`build_haca_pool.py`) drops any pool utterance sharing a word-5-gram with the frozen gold
test.

### 2.3 Heterogeneous source formats
Sources mix true SRT cues (short, mergeable) and YouTube-style paragraph dumps (long,
punctuation-free). The extractor had to window-split the long blocks and merge short cues
into ~sentence-length utterances so the quality gate has enough context.

### 2.4 Threshold calibration
Default 0.5 cutoffs are suboptimal for an imbalanced 3-class problem. Per-model calibrated
thresholds (`calibrate_thresholds.py` → `results/thresholds_<model>.json`) were needed;
the pipeline still runs without them but at lower quality.

---

## 3. Deployment-design challenges

### 3.1 A reliable system on top of an unreliable classifier
Given the ~0.5 per-utterance ceiling (§1.2), a naive auto-classifier would be a fragile
0.46. The pipeline **contours** the limitation: quality-gate abstention +
segment/programme **aggregation** (noise averages out) + **confidence and human-review
flags** (low coverage / no clear majority / low confidence → flag instead of
auto-deciding).

### 3.2 "Majority" is misleading → the non-neutral lean
A corruption documentary is ~43% negative and ~53% neutral *filler*, so plain majority
calls it "neutral". A regulator cares about the **content's valence**, so the headline
`tone` is a **non-neutral lean** gated by `NONNEU_FLOOR` (default 0.25), with raw
`majority` kept alongside for transparency.

### 3.3 Speaker tags leaking into the classifier
When per-speaker analysis was added for diarized SRTs, the `[SPEAKER_XX]` prefixes were
leaking into the text fed to the sentiment model. Fixed by stripping the tag at parse time
everywhere (which also improved the whole-programme verdict on diarized files); per-speaker
grouping merges only **consecutive same-speaker cues** so one speaker's tonality isn't
pooled with the next.

### 3.4 Multi-model maintenance vs single-model simplicity
Max precision (Option A) needs **four models** (one per language, auto-routed by language
detection) — more to maintain. The single-model option (xlm-t, F1 ~0.767) loses 7–21
points. The deployment doc lays out the trade-off rather than prescribing one answer.

---

## 4. Open problems / what would actually move the needle

1. **A bigger, balanced, multi-annotator real-domain eval set** (≥50 positives, 2
   annotators + Cohen's κ) — the single biggest lever. The current set (194 utt., 1
   annotator, 20 positives) is a pilot.
2. **Widen the source corpus beyond the 12 positive-poor SRT files** — prerequisite to
   any measurable progress on the positive class.
3. **Broadcast adaptation data** — ~300–500 broadcast examples per class to close the
   domain gap toward the 0.70 target (current ceiling ~0.52).
4. **French/arabizi broadcast coverage** — neither is represented in the real-domain set.
