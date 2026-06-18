# `haca_pipeline.py` — how the tonality pipeline works (mechanics & output)

Technical reference for the deployable system. For the *why* (design rationale), see
[DEPLOYMENT.md](DEPLOYMENT.md); this doc explains **how it runs, what it outputs, and how to read
the output**.

---

## 1. What it does (one paragraph)
It reads a broadcast SRT transcript (noisy Darija ASR) and produces a **tonality report at two
levels**: the whole **programme** and each **segment** (a sliding window of utterances). For each,
it gives the dominant tone (neg/neu/pos), the class distribution, a confidence, a **coverage**
(how much of the file was intelligible), and a **review flag** when the verdict is unreliable.
It is *not* a per-utterance classifier — it aggregates, and it abstains on noise.

---

## 2. The data flow (5 stages)

```
 SRT file
   │  ① parse + segment         (extract_utterances.load_file → build_haca_pool.utterances_for_file)
   ▼
 utterances  ──②  QUALITY GATE   (asr_quality.is_clean)  → each tagged clean | garbled
   │                              garbled = excluded from scoring (NOT labelled)
   ▼
 clean utterances ──③ CLASSIFY   (encoder pipeline + calibrated thresholds)
   │                              each clean utterance → label (neg/neu/pos) + confidence
   ▼
 ──④ AGGREGATE                    per sliding window (segment) AND over the whole file (programme)
   │                              pool labels → dominant tone, distribution, mean confidence, coverage
   ▼
 ──⑤ FLAG FOR REVIEW              low coverage / no clear majority / low confidence → flag_review=True
   ▼
 report (console timeline + JSON + dashboard CSV)
```

**Stage details**
1. **Parse + segment.** Reuses the exact same segmentation as training: long YouTube-style
   paragraphs are window-split, short SRT cues are merged into ~sentence-length utterances.
   Utterances shorter than 40 chars are dropped.
2. **Quality gate** (`asr_quality.is_clean`). Scores each utterance on function-word density,
   length, script and repetition. Garbled fragments are marked `garbled` and **kept out of the
   score** — they still count toward `n_total` (so they lower `coverage`), but they never receive
   a tone. This is the abstention that keeps precision high.
3. **Classify.** Loads `checkpoints/<model>/` as a HuggingFace text-classification pipeline and,
   if present, the calibrated thresholds from `results/thresholds_<model>.json`. Each clean
   utterance gets a label via **shifted argmax** (`score[c] − threshold[c]`) and a confidence
   (`max softmax`).
4. **Aggregate.** A segment = `WINDOW` (default 12) consecutive utterances. For each segment and
   for the whole programme, it counts the clean-utterance labels → distribution + dominant tone,
   averages the confidences, and computes coverage = `n_clean / n_total`.
5. **Flag.** `flag_review = True` if **coverage < 0.40** (too much was unintelligible) **or**
   the dominant class is **< 50 %** of clean utterances (no clear tone) **or** mean **confidence
   < 0.50**. The reason string says which.

---

## 3. The output

### 3.1 Console (quick scan)
```
8.srt           tone=NEG  props={'neg': 0.519, 'neu': 0.454, 'pos': 0.028}  conf=0.61  coverage=0.864 (108/125)
  timeline: ▼ ▼ ■ ▼ ■ ■ ■ ■ ▼ ▼ ▼    (▲pos ■neu ▼neg ·review)
```
- one line per programme: dominant tone, class proportions, confidence, coverage `(n_clean/n_total)`;
- the **timeline** = one symbol per segment in order: `▲` pos, `■` neu, `▼` neg, `·` flagged-for-review.

### 3.2 JSON (`--out file.json` or `--out-dir dir/`) — full detail
One object per file:
```json
{
  "file": "8.srt", "fmt": "youtube",
  "programme": {
    "dominant": "neg",
    "distribution": {"neg": 56, "neu": 49, "pos": 3},
    "proportions": {"neg": 0.519, "neu": 0.454, "pos": 0.028},
    "confidence": 0.61, "coverage": 0.864,
    "n_clean": 108, "n_total": 125,
    "flag_review": false, "review_reason": null
  },
  "segments": [ { "window": [0, 11], "dominant": "neg", "distribution": {...},
                  "proportions": {...}, "confidence": ..., "coverage": ...,
                  "flag_review": ..., "review_reason": ... }, ... ]
}
```

### 3.3 Dashboard CSV (`--csv file.csv`) — one row per segment/programme
Columns: `file, level, segment, dominant, p_neg, p_neu, p_pos, confidence, coverage, n_clean,
n_total, flag_review, reason`. `level` is `programme` (segment = `-`) or `segment` (segment =
`startIdx-endIdx`). **Import directly into Excel / Power BI** to build the tonality dashboard.

### 3.4 How to read a verdict
| Field | Meaning | Use |
|---|---|---|
| `dominant` | most frequent tone among **clean** utterances | the headline tonality |
| `proportions` | share of neg/neu/pos | how mixed the segment is |
| `coverage` | clean / total utterances | trust signal — low = noisy file |
| `confidence` | mean max-softmax | trust signal — low = uncertain |
| `flag_review` | needs a human | route these to a reviewer, don't auto-decide |

A **high-coverage, high-confidence, clear-majority** segment is trustworthy. A **flagged** one
goes to the review queue — that's the design, not a bug.

---

## 4. How to run

```bash
# real run (machine with the checkpoint + GPU)
python src/haca_pipeline.py --srt data/raw/srt/8.srt --model marbertv2-haca
python src/haca_pipeline.py --srt-dir data/raw/srt --model marbertv2-haca \
       --out tonality.json --csv tonality.csv

# plumbing test without a model (keyword classifier)
python src/haca_pipeline.py --srt-dir data/raw/srt --stub
```
Options: `--srt` (one file) or `--srt-dir` (a folder); `--model` (checkpoint name, default
`marbertv2-haca`); `--out` / `--out-dir` (JSON); `--csv` (dashboard); `--stub` (no model).

**Prerequisites for a real run:** `checkpoints/<model>/` must exist (train it via
`finetune.py` / the Kaggle notebook) and, ideally, `results/thresholds_<model>.json` (from
`calibrate_thresholds.py`). Without the thresholds file it still runs, using default 0.5 cuts.

> ⚠ The raw SRTs live in `data/raw/srt/`, which is **gitignored** — so a fresh clone (e.g. on
> Kaggle) has an empty folder and the pipeline finds 0 files (output `[]`). Put the SRTs there
> first (commit them, or attach them as a Kaggle dataset and point `--srt-dir` at it).

---

## 5. Tuning the review behaviour
Three constants at the top of `haca_pipeline.py` control how aggressive the abstention is:
- `WINDOW` (12) — utterances per segment (bigger = smoother, fewer segments);
- `COVERAGE_MIN` (0.40), `MAJORITY_MIN` (0.50), `CONF_MIN` (0.50) — raise them for a stricter
  system (more goes to human review, higher precision on the rest); lower them for more
  automation. Set these to match the human-review capacity HACA accepts.

---

## 6. Why this beats a per-utterance classifier
Per-utterance macro-F1 caps ~0.46–0.52 on this content (FINDINGS §8), mostly because of garbled
ASR and a tiny positive class. The pipeline doesn't fight that — it **filters** the noise and
**aggregates** to the level a regulator actually uses, so the headline verdict (e.g. a corruption
documentary → **negative**) is reliable even when individual cues are unreadable. Accuracy at the
segment level is not captured by the per-utterance number; to measure it properly, annotate a few
programmes at the segment level (see TEST_SET_PROTOCOL.md).
