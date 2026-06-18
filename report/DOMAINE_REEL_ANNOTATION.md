# Domaine-Réel Test Set — Annotation Rules & Provenance

> **Note (2026-06):** This document records the **v1** speaker-sentiment rules used to build
> `domaine_reel.csv`. The **canonical** labeling rule for all *new* HACA data
> (training pool + synthetic) is now content-valence — see
> [ANNOTATION_RUBRIC_V3.md](ANNOTATION_RUBRIC_V3.md). The frozen gold test
> `domaine_reel_v2.csv` already follows the content-valence philosophy.

**Author:** Marwane ElBaraka
**File:** `data/test_sets/domaine_reel.csv`
**Size:** 194 utterances
**Label distribution:** neg = 38 · neu = 146 · pos = 10
**SHA-256:** `412cc5b46e9269b1cf54303f7ab5670783410c6fefdb77b633d8f7f2ee33f844`

This document records how the domaine-réel ("real domain") sentiment test set was
built from actual SRT files, the rules used to assign each label, and the known
limitations of the set. It exists so the set is reproducible and its biases are
documented — a test set without written annotation rules is not trustworthy.

---

## 1. Why this set exists

Steps 2–5 of the benchmark measured every model on **public academic datasets**
(MAC, Allociné, ASTD, MYC) — social-media text that may not match the language,
register, or sentiment distribution of the **broadcast content** the models will
actually run on at HACA. This set measures the gap directly: it is built from real
`.srt` transcripts and is used for evaluation only (never training).

See [SRT_PIPELINE.md](SRT_PIPELINE.md) for the full extraction pipeline.

---

## 2. Source material

12 files in `data/raw/srt/`, of two physical kinds:

| Kind | Files | Content |
|---|---|---|
| Broadcast subtitles (true SRT) | 1, 2, 6, 7769, 7770 | TV programmes: a political debate show (*نقطة إلى السطر*, Al-Oula), a medical talk show, a religious/Sufi programme, plus two news/misc files |
| YouTube auto-transcripts (`(mm:ss)` text) | 3, 4, 5, 7, 8, 9, 10 | Long-form Darija explainer videos: 2025 tax law, public procurement, the stock market, Sahara history, corruption, Maghreb history, the health sector |

All content is **Moroccan Darija in Arabic script** (`detected_lang = darija_ar`
for every utterance after camel-tools refinement — there was no French or Arabizi in these files).

### Language routing
The first-level heuristic (`detect_lang`) returns `arabe` for any Arabic-script text.
A camel-tools `DialectIdentifier.pretrained()` pass was then run on all 194 utterances:
73% predicted Moroccan cities (RAB/FES), 7 predicted MSA, the remainder predicted Gulf
city codes (noise from garbled ASR and formal vocabulary). Because all source files are
Moroccan-origin content, the 7 spurious MSA predictions were also overridden to
`darija_ar` — including `7769.srt_0018` which contains clear Darija markers ("غتخليوني",
"ديالنا") and was clearly misclassified. Final result: all 194 utterances → `darija_ar`.

### Extraction
- `src/extract_utterances.py` parses both formats (SRT via `srt_utils.parse_srt`;
  YouTube via a `(mm:ss)` block regex), merges cues into utterances, drops
  fragments < 40 chars, truncates very long YouTube blocks to 400 chars at a word
  boundary, then draws a **per-file-balanced sample** (~16 utterances/file) so no
  single long file dominates.
- 763 candidate utterances were extracted; 194 were sampled for annotation.

---

## 3. Label definitions

Three classes, matching the benchmark's canonical scheme (`neg` / `neu` / `pos`).
The unit of judgement is the **utterance** (one merged sentence or one transcript
block), labelled by its **dominant expressed sentiment**.

### `pos` — positive
The utterance expresses praise, success, optimism, celebration, or a clearly
favourable judgement.
- *Examples:* shares that "multiplied 7×" (5.srt); "ends 141 years of humiliation
  and restores Morocco" (7.srt); the UN praising an initiative as "serious and
  realistic" (7.srt); "we enter these elections with great hope" (7769.srt).

### `neg` — negative
The utterance expresses criticism, complaint, failure, a problem, an accusation,
or a clearly unfavourable judgement.
- *Examples:* "115 billion embezzled from the social-security fund" (8.srt);
  "doctors emigrate because they hate the conditions" (10.srt); "the reforms favour
  the rich, the middle class was excluded" (3.srt); "the government is incapable"
  (7769.srt); "the middle class has been killed off — what did you do for the poor?"
  (7770.srt).

### `neu` — neutral
The utterance is factual, descriptive, procedural, or a greeting/transition with no
clear emotional charge — **even if the broader topic is sensitive.** Explaining
*how* a tax bracket works, *what* a hospital network contains, or *which* dynasty
ruled when is neutral; only an explicit value judgement moves it to neg/pos.
- *Examples:* "the income-tax table has these brackets…" (3.srt); historical
  narration of the Almoravid empire (9.srt); "welcome to a new episode" (7769.srt);
  the legal definition of corruption (8.srt).

---

## 4. Decision rules (applied consistently)

1. **Topic ≠ sentiment.** A neutral explanation of a negative *subject* (how
   corruption is defined, how a disease progresses) is `neu`. The label tracks the
   speaker's expressed stance, not the unpleasantness of the theme. This is the most
   important rule and the main reason `neu` dominates an explainer-heavy corpus.

2. **Explicit cue words decide borderline cases.** Words like *للأسف* (unfortunately),
   *مشكل* (problem), *فساد* (corruption), *حقرة* (humiliation), *مسكين* (poor/pity)
   pull toward `neg`; *فرح* (joy), *أمل* (hope), *مزيان/زوين* (good/nice), *نجاح*
   (success) pull toward `pos`.

3. **Dominant sentiment wins.** For mixed utterances, label the prevailing stance.
   A passage that lists a reform then says "but in reality workers didn't benefit"
   is `neg` because the critical clause is the point being made.

4. **Procedural / how-to content is `neu`.** The procurement and stock-market
   explainers (4.srt, 5.srt) are largely step-by-step instructions → `neu`, except
   where an explicit judgement appears (e.g. "baksheesh keeps people out" → `neg`).

5. **Greetings, intros, transitions, sponsorship reads are `neu`.**

6. **Garbled ASR fragments default to `neu`.** Files 1, 2, and 6 are poor
   speech-to-text output: many utterances are syntactically incoherent and carry no
   recoverable sentiment. These are labelled `neu` unless a clear sentiment word
   survives the noise (e.g. "مسكين" → neg, "خليهم يفرحوا" → pos). See limitations.

---

## 5. Known limitations & biases

- **Class imbalance (neu-heavy, pos-scarce).** Educational explainers and political
  critique dominate the source material, so `neu` (146) and `neg` (38) vastly
  outnumber `pos` (10). With only 10 positive examples, the per-class F1 for `pos`
  will be high-variance — treat it as indicative, not precise. This imbalance is a
  **genuine property of the corpus**, not an artefact: sincere positive sentiment is
  rare in regulatory/news/explainer broadcast content. Report macro-F1 alongside the
  per-class breakdown and state the support counts.

- **ASR noise (files 1, 2, 6).** ~36 utterances come from low-quality transcription
  and are barely intelligible. They are realistic (HACA will encounter bad ASR) but
  they make the `neu` class partly a "no-recoverable-signal" class rather than a
  "genuinely neutral statement" class. A stricter version of this set would exclude
  them; they are kept here to reflect real input conditions.

- **Single annotator.** All labels were assigned by one annotator in one pass. There
  is no inter-annotator agreement score. Borderline neu/neg calls (≈ 15–20 of them)
  are the most likely source of label noise.

- **Topic skew.** Heavy on economics/politics/governance; light on culture, sport,
  weather, human interest. Model performance here may not generalise to those.

- **Script only.** 100 % Arabic script — this set does **not** exercise the French,
  MSA-vs-Darija, or Arabizi routing branches of the pipeline.

---

## 6. How to reproduce

```bash
# 1. extract & sample utterances from data/raw/srt/*.srt
python src/extract_utterances.py --sample 200 --out data/test_sets/domaine_reel_raw.csv

# 2. apply the manual labels (kept in src/apply_annotations.py) and freeze
python src/apply_annotations.py
# → data/test_sets/domaine_reel.csv  (+ prints distribution and SHA-256)
```

The label decisions themselves live, with per-utterance justifications, in the
`LABELS` dict of `src/apply_annotations.py`.

---

## 7. How it is evaluated

The set is run through the standard harness, applying the **best model per language**
(here: Arabic-script → MARBERTv2 fine-tuned, with camel-tools MSA/Darija refinement
where available). Because the set is overwhelmingly `neu`, the headline number is
**macro-F1** (equal weight per class), not accuracy — accuracy would be inflated by a
model that simply predicts `neu` everywhere. Results and the gap-versus-public-datasets
analysis are written up in [RESULTS_ANALYSIS.md](RESULTS_ANALYSIS.md).
