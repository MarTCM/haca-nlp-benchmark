# HACA Annotation Rubric v3 — Content-Valence (canonical)

**Status:** Canonical labeling rule for all HACA sentiment data from 2026-06 onward.
**Supersedes:** the speaker-sentiment rules in `DOMAINE_REEL_ANNOTATION.md` (v1) for *new*
data. The v1/v2 frozen test sets are kept as-is for comparison.
**Applies to:** the new training pool (`haca_pool_v3.csv` → `haca_labeled_v3.csv`), the
synthetic data, and any future broadcast annotation. The frozen gold test
`domaine_reel_v2.csv` already follows this philosophy and is **not** relabeled.

---

## 1. What we are labeling

The unit is **one utterance** (a merged sentence or transcript clause). We label the
**valence of the content being reported**, *not* the presenter's tone or emotion.

> A programme that reports a corruption scandal or doctors emigrating is **negative**, even
> if the anchor stays calm and professional. HACA regulates the **content broadcast**, not
> the presenter's posture.

This is the single most important rule and the reason the scheme is called *content-valence*.

## 2. The three classes

### `neg` — the content reports something bad
Failure, loss, corruption, shortage, injustice, conflict, decline, danger, a problem, an
accusation, a broken promise, a population harmed.
- *Real examples:* "63% of health costs come out of citizens' own pockets" (10.srt);
  "115 billion embezzled from social security" (8.srt); "doctors emigrate — half of each
  graduating class leaves" (10.srt); "the reform favoured the rich, the middle class was
  excluded" (2.srt); "the government is incapable" (7769.srt).

### `pos` — the content reports something good
Success, achievement, reform that helps, growth, an opportunity the listener can seize,
progress, recovery, optimism, a benefit gained.
- *Real examples:* "shares multiplied 7× — 10k became 70k" (4.srt); "you can claim up to
  100% of your income as a pension, for life" (2.srt); "the Green March restored Morocco
  after 141 years" (7.srt); "you can win these public contracts — here's the 245-billion
  opportunity" (3.srt); "the new 2025 table means a worker on 6000 keeps more" (2.srt).

### `neu` — no recoverable valence
Procedural / definitional / how-to / descriptive / greeting / transition / sponsor read —
**regardless of how sensitive the topic is** — and ASR-garbled fragments with no recoverable
meaning.
- *Real examples:* "the income-tax table has these brackets: 0–30k → 0%, …" (2.srt); "the
  health system has three pillars: …" (10.srt); the legal definition of corruption (8.srt);
  historical narration of which dynasty ruled when (9.srt); "welcome to a new episode"
  (7769.srt); "الحلو جانيت ريكاز الريان يفاخر" (garbled, 1.srt).

## 3. Decision rules (apply in order)

1. **Topic ≠ valence.** A neutral *explanation* of a bad subject (how corruption is defined,
   how a disease spreads, how a tax bracket works) is `neu`. Only an explicit good/bad
   *outcome or judgement* moves it to `pos`/`neg`. This rule keeps `neu` honest in an
   explainer-heavy corpus.
2. **Outcome test.** Ask: "does this utterance tell me something turned out well or badly for
   someone?" Yes-bad → `neg`; yes-good → `pos`; no / just describing mechanics → `neu`.
3. **Dominant clause wins.** For mixed utterances, label the clause that is the *point*.
   "They announced a reform, **but workers didn't benefit**" → `neg` (the critique is the
   point). "There were problems, **but the new fund now covers everyone**" → `pos`.
4. **Procedural / how-to stays `neu`** unless an explicit value judgement appears
   ("baksheesh keeps capable people out" inside a procurement how-to → `neg`).
5. **Greetings, intros, transitions, sponsor reads → `neu`.**
6. **Garbled ASR → `neu`**, but flag it (`quality=garbled`); these are **quarantined from
   training** so `neu` doesn't become a "no-signal" trash class (see Stage 1).
7. **Cue words help borderline calls** (not decisive alone):
   - toward `neg`: للأسف (unfortunately), مشكل (problem), فساد (corruption), حقرة
     (humiliation), مسكين (poor/pity), خسر (lost), نقص (shortage), هاجر (emigrated).
   - toward `pos`: نجاح (success), فرح/فرحة (joy), أمل (hope), مزيان/زوين (good/nice),
     تضاعف (multiplied), استفاد (benefited), فرصة (opportunity), تحسن (improved).

## 4. Hard borderline conventions (consistency anchors)

These are the calls that caused noise in v1/v2 — fix them once, apply everywhere:

| Situation | Label | Why |
|---|---|---|
| Explaining a tax/procurement/market *mechanism* | `neu` | mechanics, no outcome |
| "This reform **helps** group X" | `pos` | explicit good outcome |
| "This reform **failed / excluded** group X" | `neg` | explicit bad outcome |
| Listing a problem then a solution that works | `pos` | solution is the point |
| Listing a reform then "but it didn't work" | `neg` | critique is the point |
| Historical narration with no judgement | `neu` | descriptive |
| Historical event framed as victory/liberation | `pos` | explicit positive framing |
| "You can do X / claim X / win X" (empowering how-to) | `pos` | opportunity offered to listener |
| Pure step-by-step "first do X, then Y" | `neu` | procedure, no valence |
| Garbled / incoherent fragment | `neu` (+`garbled`) | no recoverable signal |

## 5. Provenance & disclosure

- Training labels are **authored by Claude** (`label_source = "claude-manual"`), reading each
  utterance under this rubric. Disclosed in `HACA_DATASET.md`.
- Synthetic rows are **authored by Claude** (`synthetic = true`), written to fit this rubric.
- The evaluation gold (`domaine_reel_v2.csv`) is **human-annotated** and frozen — never
  relabeled, never trained on. All model scores are reported against it.
