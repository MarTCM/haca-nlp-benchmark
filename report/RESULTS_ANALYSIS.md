# Results Analysis — All Models (Steps 3 & 4)

**Author:** Marwane ElBaraka  
**Date:** 2026-06-16  
**Scope:** Complete analysis covering three pre-trained models (Step 3), four fine-tuned models (Step 4), and Atlas-Chat-2B LLM zero-shot (Step 5), across all four target languages. Figure analysis included at the end.

---

## Table of Contents

1. [How to read the metrics — a primer](#1-how-to-read-the-metrics--a-primer)
2. [How to read a JSON result file](#2-how-to-read-a-json-result-file)
3. [How to spot red flags from numbers alone](#3-how-to-spot-red-flags-from-numbers-alone)
4. [Results overview](#4-results-overview)
5. [Model-by-model analysis](#5-model-by-model-analysis)
6. [Language-by-language analysis](#6-language-by-language-analysis)
7. [Cross-cutting patterns](#7-cross-cutting-patterns)
8. [What the fine-tuned models actually achieved vs predictions](#8-what-the-fine-tuned-models-actually-achieved-vs-predictions)
9. [Figure analysis](#9-figure-analysis)
10. [Checklist: how to analyse a new model yourself](#10-checklist-how-to-analyse-a-new-model-yourself)

> **Step 5 addition (Atlas-Chat-2B):** Sections 4.1, 5.8, 6.1, 6.4, and 7.7 have been updated with Atlas-Chat-2B zero-shot results.

---

## 1. How to read the metrics — a primer

Before looking at any numbers, you need a firm understanding of what each metric actually measures. This section teaches the concepts from scratch. If you already know precision/recall/F1 well, you can skim it and use it as a reference.

---

### 1.1 The confusion matrix — start here, always

Everything else is derived from the confusion matrix. It is a table where:
- **Rows** = the **true** (correct) label
- **Columns** = the **predicted** label the model output

For a 3-class problem (neg / neu / pos):

```
                Predicted neg   Predicted neu   Predicted pos
True neg           TP_neg          FN_neu          FN_pos
True neu           FP_neg          TP_neu          FN_pos
True pos           FP_neg          FP_neu          TP_pos
```

Diagonal cells (top-left to bottom-right) = **correct predictions**.  
Off-diagonal cells = **errors**.

**Real example — xlm-t on darija_ar:**

```
              Predicted neg   Predicted neu   Predicted pos
True neg           169             19              13
True neu            60            120              52
True pos            37             45             485
```

Reading row by row:
- Of the 201 true-negative tweets, 169 were correctly predicted neg, 19 were predicted neu, and 13 were predicted pos.
- Of the 232 true-neutral tweets, 60 were predicted neg, 120 were predicted neu, and 52 were predicted pos. Only 120/232 = 51.7% of neutral tweets were caught.
- Of the 567 true-positive tweets, 37 were predicted neg, 45 were predicted neu, and 485 were predicted pos. 485/567 = 85.5% were caught.

The confusion matrix tells you at a glance: **which classes the model confuses with each other**. In this case, neutral is the sink for errors from both neg and pos.

---

### 1.2 Precision, Recall, and F1 — derived from the confusion matrix

For a single class X:

**Precision** = "When the model says X, how often is it right?"

```
Precision(X) = True Positives for X / (True Positives for X + False Positives for X)
             = (diagonal cell for X) / (sum of the column for X)
```

A low precision means the model over-predicts X — it labels many things as X that aren't.

**Recall** = "Of all the real X examples, how many did the model find?"

```
Recall(X) = True Positives for X / (True Positives for X + False Negatives for X)
           = (diagonal cell for X) / (sum of the row for X)
```

A low recall means the model under-predicts X — it misses many real X examples.

**F1 score** = harmonic mean of precision and recall:

```
F1(X) = 2 × Precision(X) × Recall(X) / (Precision(X) + Recall(X))
```

The harmonic mean punishes imbalance between P and R harder than the arithmetic mean would. A model with P=1.0, R=0.1 gets F1=0.18, not 0.55. This is intentional: a model that only makes 5 confident predictions and gets them all right (P=1.0, R=small) is not useful in practice.

**Worked example** from xlm-t on darija_ar:

For the `neu` class:
- The neutral column in the confusion matrix is [19, 120, 45]. Sum = 184.
- True positives = 120 (the diagonal).
- Precision(neu) = 120 / 184 = 0.652.
- The neutral row is [60, 120, 52]. Sum = 232 (= the support, the number of true neutral examples).
- Recall(neu) = 120 / 232 = 0.517.
- F1(neu) = 2 × 0.652 × 0.517 / (0.652 + 0.517) = 0.577.

These match the JSON exactly.

---

### 1.3 Macro-F1 — the headline number

Once you have an F1 for each class, **macro-F1** is the simple unweighted average:

```
Macro-F1 = (F1(neg) + F1(neu) + F1(pos)) / 3
```

For xlm-t on darija_ar: (0.724 + 0.577 + 0.868) / 3 = 0.723. ✓

**Why macro, not accuracy?**

Accuracy = (all correct predictions) / (total predictions).  
For xlm-t on darija_ar: (169 + 120 + 485) / 1000 = 0.774.

Accuracy sounds better than 0.723, but it is misleading: it is inflated by the positive class (567/1000 examples). A model that predicted everything as positive would have 56.7% accuracy while being completely wrong about neg and neu. Macro-F1 treats each class equally and exposes that failure.

**Rule of thumb for macro-F1 on a 3-class problem:**
- Random baseline: ~0.33 (33%) — predicting randomly
- Majority-class baseline: ~0.25–0.35 (below average because minority classes get F1≈0)
- Acceptable for production: ≥ 0.75
- Strong: ≥ 0.85
- Excellent: ≥ 0.92

---

### 1.4 Weighted-F1 vs Macro-F1 — knowing which to use

The JSON also reports `weighted avg F1`. This averages F1 per class, but weighted by the support (number of true examples) of each class.

```
Weighted-F1 = (F1(neg)×support(neg) + F1(neu)×support(neu) + F1(pos)×support(pos)) / total
```

For xlm-t on darija_ar: (0.724×201 + 0.577×232 + 0.868×567) / 1000 ≈ 0.772.

Weighted-F1 is closer to accuracy: it is dominated by the largest class. **Always report macro-F1 as the headline number** when classes are imbalanced or when you care equally about all classes (which we do — a false negative for neg is just as bad as one for pos).

Use weighted-F1 only as a sanity check that the most common class is not dragging the whole score down.

---

### 1.5 Latency — what the number means in practice

`latency_ms_per_utt` = milliseconds to process **one** utterance, averaged over 1000 utterances after a warm-up.

In the context of broadcast SRT files, the practical interpretation is:
- A 30-minute programme at 2 captions per second = 3600 utterances.
- At 100ms/utt (CPU), that is 360 seconds = **6 minutes** to process one programme.
- At 1000ms/utt, that is 3600 seconds = **1 hour** — unusable for near-real-time monitoring.
- At 3–4ms/utt (GPU, as achieved by fine-tuned models), that is 10–14 seconds per programme — effectively real-time.

Ready-made models in Step 3 were measured on CPU only (`peak_vram_mb = 0.0`). Fine-tuned models in Step 4 were measured on Kaggle's T4 GPU, giving 3–4ms per utterance — a 30–300× speedup depending on which CPU model you compare against.

---

## 2. How to read a JSON result file

Every `results/<model>_<lang>.json` has the same structure. Here is the template with annotations:

```json
{
  "model":   "xlm-t",           // short identifier, matches summary.csv
  "lang":    "darija_ar",       // one of: darija_ar, francais, msa, arabizi
  "n":       1000,              // test set size (always 1000 in this benchmark)
  "macro_f1": 0.723,            // THE headline number — start here

  "latency_ms_per_utt": 109.6,  // ms per utterance (CPU for Step 3, GPU for Step 4)
  "peak_vram_mb": 0.0,          // GPU memory used (0 = no GPU, >0 = Kaggle T4 GPU)
  "n_params": 278045955,        // model parameter count (278M here)

  "classes_evaluated": ["neg","neu","pos"],  // which classes exist in this test set

  "classification_report": {
    "neg": {
      "precision": 0.635,       // of predicted neg, 63.5% were truly neg
      "recall":    0.841,       // of true neg, 84.1% were caught
      "f1-score":  0.724,       // harmonic mean of P and R
      "support":   201.0        // number of true neg examples in the test set
    },
    "neu": { ... },
    "pos": { ... },

    "accuracy":  0.774,         // raw accuracy (less informative than macro-F1)
    "macro avg": { "f1-score": 0.723 },     // = macro_f1 at the top
    "weighted avg": { "f1-score": 0.772 }   // weighted by class frequency
  },

  "confusion_matrix": [
    [169, 19, 13],   // true neg row: [pred_neg, pred_neu, pred_pos]
    [ 60,120, 52],   // true neu row
    [ 37, 45,485]    // true pos row
  ],
  "confusion_matrix_labels": ["neg","neu","pos"],

  // fine-tuned model fields (Step 4 only):
  "gpu_train_minutes": 11.6,    // how long fine-tuning took on the Kaggle T4
  "checkpoint": "/kaggle/..."   // path to the saved model weights

  // Step 3 optional fields:
  "binary_original": false,   // camelbert-da was documented as binary but has 3 classes
  "note": "model has native neutral class"
}
```

**Reading order every time you open a new JSON:**

1. Check `classes_evaluated` — is this a 2-class or 3-class evaluation?
2. Read `macro_f1` — is the headline number good, bad, or mediocre?
3. Open the `classification_report` and find the class with the **lowest F1**. That is the weak point.
4. For that weak class, compare precision vs recall:
   - Low recall + high precision → model misses many real examples of this class (under-predicts)
   - Low precision + high recall → model labels many things as this class that aren't (over-predicts)
5. Open the confusion matrix. Find the row for the weak class. Which column absorbs most of its errors? That tells you what the model confuses it with.
6. Check `latency_ms_per_utt` — is the model fast enough for the deployment scenario?

---

## 3. How to spot red flags from numbers alone

These are patterns that should trigger deeper investigation:

| Pattern | What it means | Example in this benchmark |
|---|---|---|
| **High precision, low recall on a class** | Model under-predicts that class — hedges and stays silent | xlm-t on arabizi: neg P=0.921, R=0.676 |
| **Low precision, high recall on a class** | Model over-predicts that class — fires at too many inputs | camelbert-da neg on darija_ar: P=0.581, R=0.905 |
| **Neutral F1 << pos and neg F1** | Model cannot distinguish neutral from polar classes | All Arabic models in this benchmark |
| **Row sums in confusion matrix < support** | Model predicted a class not shown in the matrix (mystery predictions) | xlm-t on binary test sets (predicts "neu" where there is none) |
| **Weighted-F1 >> macro-F1** | The minority class is pulling macro-F1 down; the majority class is fine | xlm-t darija_ar: weighted=0.772, macro=0.723 |
| **macro-F1 barely above 0.33** | Model is near-random; not learning the task | Would apply if a model scored 0.40–0.50 |
| **Very high accuracy but mediocre macro-F1** | Majority class dominates; minority classes almost ignored | Would appear if model always predicted "pos" on darija_ar (57% accuracy, but neg/neu F1≈0) |

---

## 4. Results overview

### 4.1 Summary table — all models

| Model | darija\_ar | francais | msa | arabizi | Params | Type |
|---|:---:|:---:|:---:|:---:|:---:|---|
| xlm-t | 0.723 | 0.772 | 0.813 | 0.762 | 278M | Multilingual, ready-made |
| camelbert-da | 0.701 | — | **0.924** | — | 109M | Arabic-specialist, ready-made |
| distilcamembert | — | **0.949** | — | — | 68M | French-specialist, ready-made |
| DarijaBERT | 0.775 | — | — | — | 147M | Fine-tuned (Step 4) |
| DarijaBERT-arabizi | — | — | — | **0.983** | 171M | Fine-tuned (Step 4) |
| MARBERTv2 | **0.844** | — | 0.838 | — | 163M | Fine-tuned (Step 4) |
| QARIB | 0.827 | — | 0.812 | — | 135M | Fine-tuned (Step 4) |
| Atlas-Chat-2B | 0.727 | — | — | 0.978 | 1602M | LLM zero-shot (Step 5) |
| Atlas-Chat-9B | 0.833 | — | — | 0.754† | 5080M | LLM zero-shot (Step 5) |

**Bold** = best result for that language column across all models tested.  
† Atlas-Chat-9B predicts "neutre" for 42.4% of arabizi inputs (abstention on a binary test set).

Key takeaways at a glance:
- Fine-tuned models set new best scores on every language they were trained for except MSA, where the ready-made camelbert-da (0.924) remains unbeaten.
- DarijaBERT-arabizi's 0.983 is the highest single (model × language) score in the entire benchmark.
- MARBERTv2 is the strongest model for Moroccan Darija (0.844), well above the 0.75 threshold from the plan.
- Atlas-Chat-2B scores 0.727 on darija_ar but 0.978 on arabizi (0.5 pts behind DarijaBERT-arabizi).
- Atlas-Chat-9B dramatically reverses this: 0.833 on darija_ar (only 1.1 pts behind MARBERTv2, zero-shot) but collapses to 0.754 on arabizi due to 42.4% neutral abstention. Both LLMs are 45–115× slower than fine-tuned encoders.

---

### 4.2 Test set profiles (important context for interpreting results)

| Language | n | Classes | Distribution | Notes |
|---|---|---|---|---|
| darija\_ar | 1000 | 3 (neg/neu/pos) | pos=567, neu=232, neg=201 | Imbalanced; pos-heavy |
| francais | 1000 | 2 (neg/pos) | neg=520, pos=480 | Near-balanced; movie reviews |
| msa | 1000 | 2 (neg/pos) | neg=679, pos=321 | Skewed; Egyptian political Twitter |
| arabizi | 1000 | 2 (neg/pos) | pos=657, neg=343 | Pos-heavy; YouTube entertainment |

The class distribution affects which metrics matter most. On the MSA test set with 67.9% negative, a model that always predicts "neg" would score 67.9% accuracy — but only 0.40 macro-F1 (because F1(pos)≈0). Keep this in mind.

---

### 4.3 Training cost (fine-tuned models)

| Model | Languages | GPU training time | Hardware |
|---|---|---|---|
| DarijaBERT | darija\_ar | 11.6 min | Kaggle T4 |
| DarijaBERT-arabizi | arabizi | **3.4 min** | Kaggle T4 |
| MARBERTv2 | darija\_ar + msa | 11.8 min | Kaggle T4 |
| QARIB | darija\_ar + msa | 11.3 min | Kaggle T4 |

Training costs are negligible: all four models were fine-tuned on free Kaggle compute in under 12 minutes each. DarijaBERT-arabizi finished in 3.4 minutes — the shortest training run in the benchmark — and produced the highest score. This illustrates an important principle: when the training data and model pre-training are well-matched, fine-tuning converges quickly and strongly.

---

## 5. Model-by-model analysis

---

### 5.1 xlm-t (`cardiffnlp/twitter-xlm-roberta-base-sentiment`)

**What it is:** 278M-parameter multilingual model fine-tuned on 198 million tweets across 100+ languages. The only model in this benchmark that covers all four target languages without any task-specific fine-tuning.

**Headline numbers:**

| Language | Macro-F1 | Accuracy | Latency (CPU ms) |
|---|---|---|---|
| darija\_ar | 0.723 | 0.774 | 109.6 |
| francais | 0.772 | 0.771 | 1006.1 |
| msa | 0.813 | 0.767 | 168.4 |
| arabizi | 0.762 | 0.639 | 145.3 |

**Per-class breakdown:**

*darija_ar (3-class):*
```
neg:  P=0.635  R=0.841  F1=0.724  support=201
neu:  P=0.652  R=0.517  F1=0.577  support=232   ← weakest
pos:  P=0.882  R=0.855  F1=0.868  support=567
macro-F1 = 0.723
```

The model does well on positive (F1=0.868) and adequately on negative (F1=0.724), but neutral drags the average down (F1=0.577). The neutral class has low recall (51.7%): 60+52=112 neutral tweets are misclassified as neg or pos. Looking at the confusion matrix row for neu: [60, 120, 52] — 60 went to neg, 52 went to pos. The errors are split roughly evenly between the two polar classes, suggesting the model does not systematically lean one way for neutral content; it simply cannot find a reliable signal.

*francais (binary):*
```
neg:  P=0.788  R=0.765  F1=0.777  support=520
pos:  P=0.758  R=0.777  F1=0.767  support=480
macro-F1 = 0.772
```

Symmetric performance — both classes score around 0.77. Confusion matrix: [[398, 119], [107, 373]]. The model makes 119+107=226 errors out of 1000, which is 22.6% error rate. This is acceptable as a generalist baseline but poor compared to the French specialist (distilcamembert at 2.0% error rate). The main cost is that xlm-t was not specifically trained on French movie reviews and lacks the domain sensitivity that movie vocabulary requires.

**Important: the latency anomaly on francais (1006ms vs ~100–170ms on other languages).** Allociné reviews are long — a French movie review can be several paragraphs. XLM-RoBERTa truncates at 512 tokens, but the tokeniser still has to process every character of the original text before truncating. Long inputs increase tokenisation time. This latency is an artefact of the test set, not a property of the model applied to broadcast SRT captions (which are short, typically 5–15 words per cue).

*msa (binary):*
```
neg:  P=0.948  R=0.750  F1=0.837  support=679
pos:  P=0.772  R=0.804  F1=0.788  support=321
macro-F1 = 0.813
```

High precision on neg (0.948) but lower recall (0.750). The confusion matrix shows [[509, 76], [28, 258]], but these only sum to 871, not 1000. The 129 missing samples are the ones xlm-t predicted as "neutral" — a class that doesn't exist in this test set. The model is hedging 12.9% of the time on MSA content, predicting "neutral" when the ground truth is either neg or pos.

*arabizi (binary):*
```
neg:  P=0.921  R=0.676  F1=0.780  support=343
pos:  P=0.933  R=0.619  F1=0.745  support=657
macro-F1 = 0.762
accuracy = 0.639
```

The recall numbers (0.676 and 0.619) are alarming alongside the precision numbers (0.921 and 0.933). The model is confident when it makes a prediction, but it only makes predictions for 639 of the 1000 inputs — the remaining 361 are predicted as "neutral". The confusion matrix [[232, 29], [20, 407]] sums to only 688. The 312 missing samples were predicted as neutral. The model is essentially saying "I don't understand this" by defaulting to the middle class for Arabizi content.

**Summary verdict for xlm-t:** The only model that covers all four languages, making it a strong practical baseline. Its weakness is the neutral class misfire on binary test sets and the generic training that disadvantages it against specialists. Its strength is consistency — no language is catastrophically bad (all above 0.72).

---

### 5.2 camelbert-da (`CAMeL-Lab/bert-base-arabic-camelbert-da-sentiment`)

**What it is:** 109M-parameter BERT model pre-trained on dialectal Arabic (multiple Arab dialects), fine-tuned on Arabic sentiment data. The documentation listed it as binary; the actual model has 3 classes (neg/neu/pos), discovered by inspecting `id2label`.

**Headline numbers:**

| Language | Macro-F1 | Accuracy | Latency (CPU ms) |
|---|---|---|---|
| darija\_ar | 0.701 | 0.764 | 90.9 |
| msa | **0.924** | 0.893 | 163.7 |

**Per-class breakdown:**

*darija_ar (3-class):*
```
neg:  P=0.581  R=0.905  F1=0.708  support=201
neu:  P=0.708  R=0.418  F1=0.526  support=232   ← weakest
pos:  P=0.882  R=0.855  F1=0.868  support=567
macro-F1 = 0.701
```

The pattern on positive is identical to xlm-t (P=0.882, R=0.855, F1=0.868) — both models handle positive Darija similarly well.

But negative tells a very different story. Camelbert-da has **recall=0.905** on negative — it catches 90.5% of negative tweets — but **precision=0.581**, meaning many non-negative tweets are labelled as negative. Confusion matrix row for neg: [182, 9, 10] — it correctly catches 182 of 201 negative tweets (high recall). Now look at the confusion matrix column for neg: [182, 80, 51] — 80 neutral and 51 positive tweets were predicted as negative (low precision).

This is classic over-triggering: the model is biased toward predicting negative. The probable cause is that the sentiment training data was imbalanced toward negative (common in dialectal Arabic Twitter, where people often vent). The consequence: if deployed on Moroccan broadcast content (which likely includes much neutral speech), this model will generate many false negative alarms.

*msa (binary):*
```
neg:  P=0.979  R=0.895  F1=0.935  support=679
pos:  P=0.938  R=0.888  F1=0.912  support=321
macro-F1 = 0.924
```

Exceptional. Confusion matrix: [[608, 19], [13, 285]]. Total errors: 32/1000 (3.2% error rate). This is the best MSA result in the entire benchmark — including after fine-tuning.

Why does camelbert-da perform so well on MSA when it was pre-trained on dialectal Arabic? Two reasons: (1) the ASTD dataset is Egyptian Twitter, which is close to the Gulf/Egyptian dialects that dominate camelbert-da's training corpus; (2) the absence of a neutral class removes the hardest confusion — the model only has to distinguish neg from pos, which is a cleaner boundary in MSA political Twitter.

**Summary verdict for camelbert-da:** A tale of two performances. Outstanding on MSA (0.924, best in the benchmark), adequate on Darija (0.701). Smaller than xlm-t (109M vs 278M) and faster (90ms vs 110ms). Does not handle French or Arabizi.

---

### 5.3 distilcamembert (`cmarkea/distilcamembert-base-sentiment`)

**What it is:** 68M-parameter DistilCamemBERT model fine-tuned on French movie reviews (Allociné dataset), predicting star ratings 1–5 remapped to neg/neu/pos.

**Headline numbers:**

| Language | Macro-F1 | Accuracy | Latency (CPU ms) |
|---|---|---|---|
| francais | **0.949** | 0.920 | 420.9 |

**Per-class breakdown:**

*francais (binary):*
```
neg:  P=0.989  R=0.898  F1=0.942  support=520
pos:  P=0.968  R=0.944  F1=0.956  support=480
macro-F1 = 0.949
```

Confusion matrix: [[467, 15], [5, 453]]. Total errors: **20/1000** (2.0% error rate). For comparison, xlm-t makes 226 errors on the same test set. Distilcamembert makes 11× fewer mistakes.

Breaking down the 20 errors:
- 15 negative reviews predicted as positive (false positives for "pos")
- 5 positive reviews predicted as negative (false negatives for "neg")

The asymmetry (15 vs 5) means the model is slightly biased toward predicting positive — when it's uncertain, it leans positive. This makes intuitive sense: the model was trained on movie reviews, and review language tends to be more effusive in positive descriptions.

Despite being the smallest model at 68M parameters, distilcamembert beats both larger models by a large margin on French. This is the clearest demonstration in this benchmark of the principle: **a small specialist beats a large generalist on the specialist's home turf**.

**Summary verdict for distilcamembert:** The best performing ready-made model in this benchmark. If a production system only needs French, this is the obvious choice — highest F1, smallest size, clear domain match. The caveat is domain shift: Allociné reviews are written opinions about films, while broadcast SRT captions are spoken, often neutral or descriptive. Real-world performance on HACA's actual media content may be lower than 0.949.

---

### 5.4 DarijaBERT — fine-tuned on Moroccan Darija

**What it is:** 147M-parameter BERT model pre-trained exclusively on Moroccan Darija text, then fine-tuned on the MAC sentiment dataset (Arabic script). The only model in this benchmark pre-trained on Moroccan Darija specifically (not generic dialectal Arabic). Training time: 11.6 minutes on Kaggle T4.

**Headline numbers:**

| Language | Macro-F1 | Accuracy | Latency (GPU ms) |
|---|---|---|---|
| darija\_ar | 0.775 | 0.813 | 2.7 |

**Per-class breakdown:**

```
neg:  P=0.718  R=0.771  F1=0.743  support=201
neu:  P=0.685  R=0.694  F1=0.690  support=232   ← weakest (but much improved)
pos:  P=0.905  R=0.877  F1=0.891  support=567
macro-F1 = 0.7745
```

**What improved compared to the baselines:**

The most striking gain is in the **neutral class**. Comparing neutral F1 across all Darija models:

| Model | Neutral F1 | Neutral Recall |
|---|---|---|
| camelbert-da (ready-made) | 0.526 | 0.418 |
| xlm-t (ready-made) | 0.577 | 0.517 |
| DarijaBERT (fine-tuned) | **0.690** | **0.694** |

Neutral recall improved from 41.8% (camelbert-da) and 51.7% (xlm-t) to **69.4%** — a massive jump. This is exactly what was expected: Moroccan-specific pre-training gives the model vocabulary and phrasing patterns that are specific to neutral Moroccan content (factual statements, descriptions, questions without strong sentiment). The model can now identify the absence of polar language, not just the presence of it.

**Confusion matrix:** [[155, 30, 16], [35, 161, 36], [26, 44, 497]]

Reading the neutral row [35, 161, 36]: 35 neutral tweets went to neg, 36 went to pos. This is much more balanced than xlm-t's [60, 120, 52] — the absolute error counts in every off-diagonal cell are smaller. The total neutrals correctly classified jumped from 120 (xlm-t) to 161.

**What did not improve as much:**

Negative precision is still the weak point. DarijaBERT's neg precision (0.718) is better than xlm-t (0.635) but still noticeably below its own recall (0.771). The neg column shows [155, 35, 26] — 35 neutrals and 26 positives were predicted as negative. The negative over-triggering is much reduced (camelbert-da had 80+51=131 false negatives; DarijaBERT has 35+26=61), but the bias persists.

**Latency:** 2.7ms/utterance on GPU (T4). Compare to 109ms on CPU (xlm-t). A 30-minute programme at 2 captions/second (3600 utterances) takes under 10 seconds at GPU speed. DarijaBERT is operationally real-time.

**Plan threshold check:** The plan set ≥0.75 as the go/no-go threshold. DarijaBERT scores **0.7745** — this just barely exceeds the threshold. It clears it, but only by a small margin. MARBERTv2 and QARIB exceed it far more comfortably.

**Summary verdict for DarijaBERT:** Meets the minimum bar (0.775 > 0.75), delivers the most important improvement (neutral F1: +0.113 over xlm-t), and is 40× faster than the CPU baseline. However, it is strictly dominated by both MARBERTv2 and QARIB on Darija. Its relevance is mainly as a proof that Moroccan-specific pre-training helps even when the architecture is smaller than competitors.

---

### 5.5 DarijaBERT-arabizi — fine-tuned on Moroccan Arabizi

**What it is:** 171M-parameter BERT model pre-trained on Moroccan Arabizi (Latin-script Moroccan Arabic), fine-tuned on the MYC dataset (YouTube comments). The only model in this benchmark — indeed, likely one of the very few in existence — designed specifically for Moroccan Arabizi. Training time: **3.4 minutes** on Kaggle T4.

**Headline numbers:**

| Language | Macro-F1 | Accuracy | Latency (GPU ms) |
|---|---|---|---|
| arabizi | **0.983** | 0.985 | 3.9 |

**Per-class breakdown:**

```
neg:  P=0.977  R=0.980  F1=0.978  support=343
pos:  P=0.989  R=0.988  F1=0.989  support=657
macro-F1 = 0.9834
```

**Confusion matrix:** [[336, 7], [8, 649]]

Total errors: **15 out of 1000** (1.5% error rate). This is the single best result in the entire benchmark.

Reading the matrix:
- Of 343 true-negative Arabizi comments: 336 correctly classified, 7 incorrectly sent to positive.
- Of 657 true-positive Arabizi comments: 649 correctly classified, 8 incorrectly sent to negative.

The errors are nearly symmetric (7 neg→pos, 8 pos→neg), meaning the model has no systematic bias — it is occasionally uncertain in both directions but has no preferred direction. The decision boundary sits cleanly between the two classes.

**Why is this score so high?**

Three factors converge:
1. **Perfect pre-training match:** DarijaBERT-arabizi was pre-trained on Moroccan Arabizi text. The MYC fine-tuning data is also Moroccan Arabizi. The vocabulary, spelling conventions, and code-switching patterns are all in-domain. There is almost no distribution shift.
2. **Binary task is inherently easier:** The arabizi test set has only two classes (neg/pos) and no neutral. The model never has to make the harder three-way decision.
3. **Arabizi sentiment tends to be expressive:** YouTube comments in Arabizi often use strong, unambiguous sentiment language — many exclamation marks, intensifiers, and emotion words. This makes the neg/pos boundary sharper than in formal text.

**Compared to xlm-t on the same test set:**

| Metric | xlm-t (ready-made) | DarijaBERT-arabizi (fine-tuned) | Change |
|---|---|---|---|
| Macro-F1 | 0.762 | **0.983** | **+0.221** |
| Accuracy | 0.639 | **0.985** | **+0.346** |
| Neg recall | 0.676 | **0.980** | **+0.304** |
| Pos recall | 0.619 | **0.988** | **+0.369** |

The accuracy jump (+34.6 percentage points) is dramatic. Most of that gain comes from eliminating xlm-t's structural abstention problem: xlm-t predicted "neutral" for 312 of 1000 inputs (31.2%), absorbing all those samples as errors. DarijaBERT-arabizi has no neutral class to hide behind — it commits to neg or pos every time, and it is almost always right.

**Training cost insight:** This model trained in 3.4 minutes and achieved 0.983 macro-F1. This is among the best cost-to-performance ratios possible in NLP. The short training time reflects how well-aligned the pre-training and fine-tuning data are — the model's weights barely need to move from their pre-trained position.

**Summary verdict for DarijaBERT-arabizi:** The clear winner for Moroccan Arabizi and the highest-scoring model in the benchmark. It resolves what was the biggest open problem identified in Step 3 (the xlm-t abstention problem on Arabizi) completely. If the benchmark had a single MVP, this is it.

---

### 5.6 MARBERTv2 — fine-tuned on darija_ar and msa

**What it is:** 163M-parameter BERT model pre-trained on 1 billion words from 25 Arabic dialects and MSA, including significant Moroccan content, fine-tuned on MAC (darija_ar) and evaluated also on ASTD (msa). Training time: 11.8 minutes on Kaggle T4. Note that only one checkpoint exists — the model was fine-tuned once on Darija data and evaluated on both languages.

**Headline numbers:**

| Language | Macro-F1 | Accuracy | Latency (GPU ms) |
|---|---|---|---|
| darija\_ar | **0.844** | 0.874 | 2.5 |
| msa | 0.838 | 0.761 | 4.4 |

**Per-class breakdown — darija_ar (3-class):**

```
neg:  P=0.848  R=0.861  F1=0.854  support=201
neu:  P=0.800  R=0.707  F1=0.751  support=232   ← still weakest, but strong
pos:  P=0.909  R=0.947  F1=0.927  support=567
macro-F1 = 0.8441
```

MARBERTv2 is **the best model for Moroccan Darija** in this benchmark. The macro-F1 of 0.844 is +12.1 points over xlm-t (0.723) and +6.9 points over DarijaBERT (0.775). Every single per-class F1 score is the highest seen for darija_ar:

| Class | xlm-t | camelbert-da | DarijaBERT | MARBERTv2 |
|---|---|---|---|---|
| neg F1 | 0.724 | 0.708 | 0.743 | **0.854** |
| neu F1 | 0.577 | 0.526 | 0.690 | **0.751** |
| pos F1 | 0.868 | 0.868 | 0.891 | **0.927** |

The neutral F1 of **0.751** is the first time any model in this benchmark breaks 0.75 on the neutral class — a milestone given that it was identified in Step 3 as the hardest class for all models.

**Confusion matrix for darija_ar:** [[173, 18, 10], [24, 164, 44], [7, 23, 537]]

Reading the neutral row [24, 164, 44]: 24 neutral tweets incorrectly sent to neg, 44 to pos. These numbers are much smaller than in all previous models. The neutral diagonal (164) is the highest correct neutral count in the benchmark. The main remaining error for neutral is that 44 neutral tweets are predicted as positive — the positive class still acts as a gravitational pull for ambiguous content.

**Per-class breakdown — msa (binary):**

```
neg:  P=0.979  R=0.744  F1=0.845  support=679
pos:  P=0.865  R=0.798  F1=0.830  support=321
macro-F1 = 0.8375
```

**Confusion matrix for msa:** [[505, 40], [11, 256]]  
Matrix sum: 812. Missing: **188 samples predicted as "neutral"** despite the test set being binary.

This is the key finding for MARBERTv2 on MSA: the model learned a 3-class head during fine-tuning on darija_ar (which has neg/neu/pos), and when it encounters MSA content, it predicts "neutral" for 188 samples (18.8%) instead of committing to neg or pos. This directly lowers recall — neg recall is only 0.744 because 134 of the 679 negative samples were absorbed into the "neutral" abstention bucket.

The result (0.838) is **below camelbert-da's 0.924 on the same test set**. This is not a failure of MARBERTv2 as a model — it is a consequence of the fine-tuning setup: the model was optimised on Darija data (3-class), not MSA data. When applied to MSA, it uses the Darija-trained head, which has a neutral class that doesn't belong in the MSA evaluation.

**Summary verdict for MARBERTv2:** The strongest model for Moroccan Darija (0.844), and the first model in this benchmark to bring neutral F1 above 0.75. On MSA, it falls short of camelbert-da because of the 3-class head mismatch — the neutral class fires on MSA content that should be labelled neg or pos. If a production system routes Darija content to MARBERTv2, it is the recommended choice for that language.

---

### 5.7 QARIB — fine-tuned on darija_ar and msa

**What it is:** 135M-parameter BERT model with Gulf Arabic pre-training focus, fine-tuned on MAC (darija_ar) and evaluated on both darija_ar and msa. Training time: 11.3 minutes on Kaggle T4. Same single-checkpoint setup as MARBERTv2 — one model, two evaluation languages.

**Headline numbers:**

| Language | Macro-F1 | Accuracy | Latency (GPU ms) |
|---|---|---|---|
| darija\_ar | 0.827 | 0.853 | 2.7 |
| msa | 0.812 | 0.736 | 4.5 |

**Per-class breakdown — darija_ar (3-class):**

```
neg:  P=0.840  R=0.836  F1=0.838  support=201
neu:  P=0.721  R=0.746  F1=0.733  support=232
pos:  P=0.914  R=0.903  F1=0.909  support=567
macro-F1 = 0.8265
```

QARIB is the second-best model for Moroccan Darija (0.827), behind MARBERTv2 (0.844) but well ahead of DarijaBERT (0.775). All three per-class F1 scores are above 0.73, which is the most balanced performance profile seen on darija_ar.

Compared to MARBERTv2, QARIB has:
- Slightly lower neg F1 (0.838 vs 0.854)
- Slightly lower neu F1 (0.733 vs 0.751)
- Slightly lower pos F1 (0.909 vs 0.927)

In other words, QARIB is uniformly about 1.5–2 points below MARBERTv2 on every class. This suggests MARBERTv2's broader dialectal Arabic training (25 dialects including more Moroccan data) gives it a consistent edge over QARIB's Gulf-focused pre-training.

**Confusion matrix for darija_ar:** [[168, 23, 10], [21, 173, 38], [11, 44, 512]]

Reading the neutral row [21, 173, 38]: 21 neutral tweets went to neg, 38 to pos. Very similar to MARBERTv2's [24, 164, 44] — both models make the same type of errors (neutral → pos is the biggest error), just at slightly different magnitudes.

**Per-class breakdown — msa (binary):**

```
neg:  P=0.972  R=0.723  F1=0.829  support=679
pos:  P=0.828  R=0.763  F1=0.794  support=321
macro-F1 = 0.8118
```

**Confusion matrix for msa:** [[491, 51], [14, 245]]  
Matrix sum: 801. Missing: **199 samples predicted as "neutral"**.

Essentially the same problem as MARBERTv2 on MSA: the 3-class fine-tuning head predicts neutral for 19.9% of MSA inputs. This gives neg recall of only 0.723 (491 of 679 neg samples correctly labelled; the other 174 were predicted as neutral or positive). Pos recall is 0.763.

The QARIB MSA result (0.812) is slightly below MARBERTv2's (0.838), suggesting that MARBERTv2's broader Arabic pre-training helps even on out-of-distribution MSA content.

**Summary verdict for QARIB:** A strong model for Moroccan Darija (0.827, second-best), with the same MSA abstention problem as MARBERTv2. It is smaller than MARBERTv2 (135M vs 163M), slightly faster, and slightly lower accuracy. QARIB's main advantage would be if deployment cost (memory/compute) is a constraint — otherwise MARBERTv2 is the better choice on every language.

---

### 5.8 Atlas-Chat-2B — LLM zero-shot on darija_ar and arabizi

**What it is:** 1.6B-parameter causal language model from MBZUAI-Paris, based on Gemma-2B, pre-trained on Arabic and Moroccan content (including Darija). Evaluated zero-shot — no task-specific fine-tuning — using a closed-vocabulary Arabic prompt:

```
أنت نظام تحليل مشاعر. صنّف النص التالي إلى فئة واحدة فقط من:
positif, neutre, negatif.
النص: {text}
الإجابة (كلمة واحدة فقط):
```

The model is loaded in 4-bit NF4 quantisation on T4, consuming ~4GB VRAM. Inference is sequential (one `generate()` call per utterance, not batched). Non-response rate = fraction of inputs where the model did not output one of the three valid tokens.

**Headline numbers:**

| Language | Macro-F1 | Accuracy | Latency (GPU ms) | Non-response |
|---|---|---|---|---|
| darija\_ar | 0.727 | 0.813 | 186.4 | 0.5% (5/1000) |
| arabizi | 0.978 | 0.975 | 184.5 | 1.2% (12/1000) |

---

**Per-class breakdown — darija_ar (3-class):**

```
neg:  P=0.675  R=0.950  F1=0.789  support=201   ← over-triggers negative
neu:  P=0.894  R=0.328  F1=0.480  support=232   ← catastrophically low recall
pos:  P=0.864  R=0.963  F1=0.911  support=567
macro-F1 = 0.727
```

**Confusion matrix for darija_ar:** [[191, 0, 10], [80, 76, 76], [12, 9, 546]]

The defining feature of Atlas-Chat on 3-class darija_ar is the **collapse of neutral recall to 32.8%**. Reading the neutral row [80, 76, 76]: of 232 true-neutral tweets, 80 went to neg, 76 to pos, and only 76 were correctly identified as neutral. The errors are split almost evenly between neg and pos — the model cannot reliably identify the absence of strong sentiment.

Notice also a striking detail: the neutral column for neg is [191, 80, 12], meaning the neg-prediction column has 80 false positives from neutral. Atlas-Chat over-triggers negative: it labels 80 neutral tweets as negative, giving neg very high recall (95.0%) but low precision (67.5%). The opposite happens for positive — it absorbs 76 neutral tweets correctly but also mis-attracts them. The result is a model that handles polar sentiment well but treats neutrality as a residual category.

The non-response rate (0.5%) is negligible: only 5 inputs produced output outside `{positif, neutre, negatif}`. These were mapped to "neu" (fallback) for metric computation.

**Why does the LLM fail on neutral for Darija?**

Two compounding factors:

1. **LLM training bias:** Large language models are trained predominantly on texts where sentiment is expressed. Neutral, factual, or descriptive content appears in training data but is not inherently associated with sentiment labels. The model's prompt asks for a three-way classification, but its prior probability for "neutre" given Darija content may be intrinsically lower than for encoders that were directly supervised on neutral examples.

2. **Prompt-completion dynamics:** Causal LLMs predict the next token greedily (do_sample=False). Given the Arabic prompt ending in "الإجابة (كلمة واحدة فقط):", the most probable completion is whichever sentiment word is most strongly associated with the input tokens. For ambiguous or low-affect Darija text, the model has to make a binary-ish choice rather than confidently outputting "neutre". Encoder models have a dedicated neutral output neuron trained directly on neutral examples — this makes them structurally better equipped for the three-way distinction.

**Comparison to fine-tuned encoders on darija_ar:**

| Model | F1(neg) | F1(neu) | F1(pos) | Macro-F1 |
|---|---|---|---|---|
| xlm-t (ready-made) | 0.724 | 0.577 | 0.868 | 0.723 |
| Atlas-Chat-2B (LLM) | 0.789 | **0.480** | 0.911 | 0.727 |
| DarijaBERT (fine-tuned) | 0.743 | 0.690 | 0.891 | 0.775 |
| MARBERTv2 (fine-tuned) | **0.854** | **0.751** | **0.927** | **0.844** |

Atlas-Chat gets better per-class scores on neg and pos than xlm-t, but its catastrophic neutral F1 (0.480 vs xlm-t's 0.577) drags the macro average to near-baseline level. It is a worse model for 3-class Darija than even xlm-t.

**Plan threshold check for darija_ar:** The plan states to prefer a fine-tuned encoder over the LLM if the encoder macro-F1 ≥ 0.75 **or** if the LLM–encoder gap < 5 F1 points. MARBERTv2 scores 0.844 (≥ 0.75 ✓) and the gap is 11.7 points (not < 5 ✗ for the second condition). Both conditions together mean: **prefer the fine-tuned encoder for darija_ar. The LLM is not recommended for this language.**

---

**Per-class breakdown — arabizi (binary):**

```
neg:  P=0.982  R=0.950  F1=0.966  support=343
pos:  P=0.992  R=0.988  F1=0.990  support=657
macro-F1 = 0.978
```

**Confusion matrix for arabizi:** [[326, 5], [6, 649]]

Total errors: **11 out of 1000** (1.1% error rate, after accounting for 12 non-responses mapped to "neu" which become misclassifications). The errors are symmetric (5 neg→pos, 6 pos→neg) — no systematic directional bias. The model has a clear, confident decision boundary between negative and positive Arabizi content.

This is an exceptional result. Atlas-Chat achieves 0.978 zero-shot — without any sentiment-specific training — on a task that took DarijaBERT-arabizi 3.4 minutes of supervised fine-tuning to reach 0.983. The gap between the LLM and the best encoder is **0.5 F1 points**, well within margin of noise.

**Why does the LLM succeed on arabizi when it fails on darija_ar neutral?**

The binary nature of the arabizi task is the key difference. The model only has to choose between two polar classes (negative vs positive), which plays to the strengths of an LLM trained on human-written sentiment content. Moroccan Arabizi YouTube comments tend to use strong, unambiguous sentiment language — intensifiers, exclamation marks, emotion words — making the boundary especially sharp. There is no neutral class to get confused by.

**Plan threshold check for arabizi:** Encoder score = 0.983 (≥ 0.75 ✓). LLM–encoder gap = 0.5 points (< 5 ✓). First condition alone is sufficient: **prefer the fine-tuned encoder**. However, the gap is so small (0.5 points) that in a zero-GPU deployment scenario (no T4 available for fine-tuned inference), Atlas-Chat's zero-shot result would be fully acceptable.

**Latency reality check:**

Atlas-Chat runs at ~185ms/utterance on a T4 GPU with 4-bit quantisation. This is sequential — `generate()` is called once per text. Compare:

| Category | Latency | 30-min programme |
|---|---|---|
| Fine-tuned encoders (GPU T4) | 2–4 ms/utt | 7–14 seconds |
| Atlas-Chat-2B (GPU T4, sequential) | ~185 ms/utt | ~11 minutes |

The LLM is 45–75× slower than fine-tuned encoders on the same hardware. For near-real-time broadcast monitoring, this makes Atlas-Chat impractical even with a GPU. Its use case would be: offline batch analysis where GPU is available, and no training data or time exists for fine-tuning.

**Summary verdict for Atlas-Chat-2B:** A tale of two tasks. On binary arabizi sentiment, it is outstanding (0.978 zero-shot, 0.5 points behind the fine-tuned specialist) — proof that a well-trained Arabic LLM understands Moroccan Arabizi sentiment with no task-specific training. On 3-class darija_ar, it fails to detect neutral (F1=0.480) and offers no advantage over xlm-t. Its 185ms latency makes it 45–75× slower than fine-tuned encoders and operationally unsuitable for real-time monitoring. The fine-tuned encoders remain the recommended choice for both languages.

---

### Atlas-Chat-9B — the scale reversal

**Headline numbers:**

| Language | Macro-F1 | Accuracy | Latency (GPU ms) | Non-response | Neutral abstention |
|---|---|---|---|---|---|
| darija\_ar | **0.833** | 0.855 | 451.0 | 0.0% | — |
| arabizi | 0.754 | 0.570 | 451.3 | 0.0% | **42.4%** (424/1000) |

**Per-class breakdown — darija_ar (3-class):**

```
neg:  P=0.786  R=0.915  F1=0.846  support=201
neu:  P=0.733  R=0.759  F1=0.746  support=232
pos:  P=0.941  R=0.873  F1=0.906  support=567
macro-F1 = 0.833
```

**Confusion matrix for darija_ar:** [[184, 9, 8], [33, 176, 23], [17, 55, 495]]

This is the most surprising result in the benchmark. Atlas-Chat-9B, with zero task-specific training, scores **0.833** — only 1.1 points below MARBERTv2 (0.844), the best fine-tuned encoder for Moroccan Darija.

The neutral class improvement is the key driver. Going from 2B to 9B:

| Metric | Atlas-Chat-2B | Atlas-Chat-9B | MARBERTv2 (FT) |
|---|---|---|---|
| F1(neg) | 0.789 | 0.846 | **0.854** |
| F1(neu) | 0.480 | 0.746 | **0.751** |
| F1(pos) | 0.911 | 0.906 | **0.927** |
| macro-F1 | 0.727 | 0.833 | **0.844** |

The 9B model's neutral F1 jumps from 0.480 to **0.746** — nearly identical to MARBERTv2's 0.751. The confusion matrix neutral row [33, 176, 23] shows 176 of 232 neutral tweets correctly identified (75.9% recall), compared to only 76/232 (32.8%) for the 2B. The 9B has learned that Moroccan Arabic neutral content is genuinely neutral, not just a residual catch for ambiguous positive or negative signals.

Non-response rate is 0.0% — the 9B model always outputs one of the three valid labels. It commits confidently to all three classes on darija_ar.

**Per-class breakdown — arabizi (binary):**

```
neg:  P=0.983  R=0.863  F1=0.919  support=343
pos:  P=0.996  R=0.417  F1=0.588  support=657
macro-F1 = 0.754
```

**Confusion matrix for arabizi:** [[296, 1], [5, 274]] — sums to 576 out of 1000.

The 424 missing samples were predicted as "neutre" — a valid label in the model's vocabulary, but absent from the arabizi ground truth. Non-response rate is 0.0% because the model correctly outputs a word from `{positif, neutre, negatif}`; it just chooses "neutre" for 42.4% of inputs.

The result is catastrophic for positive recall: the model correctly identifies only 274 of 657 positive Arabizi comments (R=0.417). The other 383 are absorbed into the "neutre" bucket.

This is a direct reversal of the 2B pattern:

| | Atlas-Chat-2B arabizi | Atlas-Chat-9B arabizi |
|---|---|---|
| Neutral predictions | 12 (1.2%, non-response) | 424 (42.4%, valid label) |
| Pos recall | **0.988** | 0.417 |
| Macro-F1 | **0.978** | 0.754 |

**Why does scaling hurt on arabizi?**

The 9B model has a much stronger and better-calibrated sense of "neutrality." On the 3-class darija_ar task this is a feature: it correctly withholds polar labels from ambiguous content. But the arabizi test set is binary (neg/pos only) — there is no neutral ground truth. When the 9B sees a mildly positive Arabizi YouTube comment, it judges it as insufficiently polar and outputs "neutre." The 2B, having a weaker neutral prior, rounds to the nearest polar class and guesses correctly.

This is a form of miscalibration: the 9B is more "honest" about sentiment ambiguity, but honesty is penalised when the evaluation set has no neutral class.

**The scale reversal in full:**

| Language | 2B macro-F1 | 9B macro-F1 | Δ | Winner |
|---|---|---|---|---|
| darija_ar (3-class) | 0.727 | **0.833** | +0.106 | 9B |
| arabizi (binary) | **0.978** | 0.754 | -0.224 | 2B |

The right LLM to use depends entirely on the task structure. If the downstream task is 3-class (includes neutral), the 9B is far better. If the task is strictly binary (polar only), the 2B is far better. Neither LLM is recommended over the fine-tuned encoders for production, but this inversion is one of the most instructive findings in the benchmark.

**Plan threshold check for 9B:**
- darija_ar: encoder MARBERTv2 = 0.844 (≥ 0.75 ✓); gap = 1.1 pts (< 5 ✓) → prefer encoder, but the gap is very small
- arabizi: encoder DarijaBERT-arabizi = 0.983 (≥ 0.75 ✓); gap = 22.9 pts → prefer encoder decisively

**Summary verdict for Atlas-Chat-9B:** The benchmark's most counter-intuitive result. Scaling from 2B to 9B parameters makes the LLM nearly competitive with the best fine-tuned encoder on 3-class Darija (0.833 vs MARBERTv2 0.844, zero-shot) — but destroys its arabizi performance through neutral over-prediction (0.754 vs 2B's 0.978). The 9B takes 2.4× longer per utterance (451ms vs 184ms). For Darija, a zero-shot LLM at 0.833 with no training is a remarkable result; for production it is still 1.1 points below the fine-tuned encoder at 180× the latency.

---

## 6. Language-by-language analysis

---

### 6.1 Moroccan Darija — Arabic script (`darija_ar`)

**Rankings after fine-tuning and LLM evaluation:**

| Model | Macro-F1 | Type |
|---|---|---|
| MARBERTv2 | **0.844** | Fine-tuned |
| Atlas-Chat-9B | 0.833 | LLM zero-shot |
| QARIB | 0.827 | Fine-tuned |
| DarijaBERT | 0.775 | Fine-tuned |
| Atlas-Chat-2B | 0.727 | LLM zero-shot |
| xlm-t | 0.723 | Ready-made |
| camelbert-da | 0.701 | Ready-made |

Atlas-Chat-9B (zero-shot) ranks second — above QARIB and all other fine-tuned models except MARBERTv2. The gap between the best fine-tuned encoder and the best LLM is only 1.1 points. The three fine-tuned models all beat both ready-made baselines; the LLM 9B sits between the fine-tuned group and the ready-made group. The gap is substantial: MARBERTv2 is +12.1 points over xlm-t and +14.3 points over camelbert-da. Fine-tuning on task-specific Moroccan Darija data is clearly the right approach for this language.

**The neutral class story:**

The neutral class (F1 progression across models):
```
camelbert-da: 0.526  →  xlm-t: 0.577  →  DarijaBERT: 0.690  →  QARIB: 0.733  →  MARBERTv2: 0.751
```

This is a clear monotonic improvement as models become more Moroccan-Arabic-specific. MARBERTv2 is the first to break 0.75 on neutral — a benchmark goal that was described in Section 7 of the previous version as "outstanding if achieved". It was achieved.

**Remaining challenge:** Even at 0.751, neutral is still the weakest class by 10–17 points compared to neg and pos in the best models. Neutral content remains genuinely hard. A realistic ceiling for neutral F1 with these architectures might be around 0.80–0.82.

**Recommendation:** For production deployment on Moroccan Darija, MARBERTv2 is the clear choice. It combines the highest F1, the best neutral detection, and GPU-native speed (2.5ms/utterance).

---

### 6.2 French (`francais`)

**Rankings (unchanged — no fine-tuned French model):**

| Model | Macro-F1 | Type |
|---|---|---|
| distilcamembert | **0.949** | Ready-made |
| xlm-t | 0.772 | Ready-made |

No French model was fine-tuned in Step 4. distilcamembert remains the clear best with a +0.177 gap over xlm-t.

The pattern from Step 3 still holds: a domain-specific specialist (fine-tuned on exactly the same Allociné data as the test set) beats a 4× larger generalist by 17.7 points. If a Step 5 French-focused fine-tuning run were done on CamemBERT or a similar French-native model, it would likely score similarly to or above distilcamembert.

---

### 6.3 Modern Standard Arabic (`msa`)

**Rankings after fine-tuning:**

| Model | Macro-F1 | Type |
|---|---|---|
| camelbert-da | **0.924** | Ready-made |
| MARBERTv2 | 0.838 | Fine-tuned (Darija head) |
| xlm-t | 0.813 | Ready-made |
| QARIB | 0.812 | Fine-tuned (Darija head) |

**The surprising result:** The ready-made camelbert-da (0.924) still beats both fine-tuned models on MSA, even though MARBERTv2 and QARIB were specifically fine-tuned for this benchmark.

The reason is the fine-tuning setup. Both MARBERTv2 and QARIB were fine-tuned on MAC darija_ar data, which has 3 classes (neg/neu/pos). When they are then applied to the MSA test set (binary: neg/pos), the 3-class head predicts "neutral" for 18–20% of inputs, absorbing those as errors. This is a structural mismatch, not a learning failure.

A proper fix would be to fine-tune a separate head specifically on MSA data (ASTD) with a binary neg/pos output. That was not done in Step 4 because the checkpoint structure uses one model per pre-trained base, not one per evaluation language. This is an important architectural decision to revisit.

**Bottom line:** For MSA, camelbert-da (ready-made) is still the recommended model. The fine-tuned models lose on MSA specifically because of a 3-class → 2-class mismatch that could be resolved by a dedicated MSA fine-tuning run.

---

### 6.4 Moroccan Arabizi (`arabizi`)

**Rankings after fine-tuning and LLM evaluation:**

| Model | Macro-F1 | Type | Neutral abstention |
|---|---|---|---|
| DarijaBERT-arabizi | **0.983** | Fine-tuned | 0% |
| Atlas-Chat-2B | 0.978 | LLM zero-shot | 1.2% |
| xlm-t | 0.762 | Ready-made | 31.2% |
| Atlas-Chat-9B | 0.754 | LLM zero-shot | **42.4%** |

DarijaBERT-arabizi resolves the biggest open problem from Step 3 completely. xlm-t was the only ready-made model available for Arabizi, scoring 0.762 with 31.2% abstentions predicted as neutral. DarijaBERT-arabizi scores 0.983 with 1.5% total errors and no abstentions.

Atlas-Chat-2B zero-shot reaches 0.978 — only 0.5 points behind DarijaBERT-arabizi. This is the one case in the entire benchmark where the LLM competes with a fine-tuned encoder.

Atlas-Chat-9B ranks **last** at 0.754 — below even xlm-t. The 9B model's stronger neutral prior causes it to predict "neutre" for 42.4% of inputs, gutting positive recall to 0.417. Scaling the LLM up makes arabizi performance dramatically worse.

**Recommendation:** DarijaBERT-arabizi is the clear choice for production (45× faster than Atlas-Chat-2B). Atlas-Chat-2B is a valid fallback for offline scenarios where no GPU is available for serving a fine-tuned model. Atlas-Chat-9B should not be used for binary Arabizi sentiment without a calibration step to suppress neutral predictions.

---

## 7. Cross-cutting patterns

---

### 7.1 The neutral class is the hardest — but fine-tuning closes the gap significantly

Before fine-tuning (Step 3 baselines), neutral F1 was stuck between 0.526 and 0.577 on darija_ar. After fine-tuning (Step 4):

| Model | F1(neg) | F1(neu) | F1(pos) | Macro-F1 |
|---|---|---|---|---|
| xlm-t (ready-made) | 0.724 | 0.577 | 0.868 | 0.723 |
| camelbert-da (ready-made) | 0.708 | 0.526 | 0.868 | 0.701 |
| DarijaBERT (fine-tuned) | 0.743 | 0.690 | 0.891 | 0.775 |
| QARIB (fine-tuned) | 0.838 | 0.733 | 0.909 | 0.827 |
| MARBERTv2 (fine-tuned) | **0.854** | **0.751** | **0.927** | **0.844** |

The neutral F1 progression (0.526 → 0.577 → 0.690 → 0.733 → 0.751) shows that each step of specialisation — from generic multilingual, to Arabic dialectal, to Moroccan-specific, to Moroccan-specific + MAC fine-tuning — brings measurable improvement on neutral. Neutral remains the weakest class throughout, but the floor has been lifted from ~0.53 to ~0.75. The ceiling appears to be around 0.75–0.80 with these model sizes and training data volumes.

---

### 7.2 Language specialists beat generalists — now confirmed for Moroccan Arabic too

In Step 3, this pattern held for French and MSA but was reversed for Darija (camelbert-da was slightly worse than xlm-t). Step 4 confirms it holds for Darija as well, once the right specialist exists:

| Comparison | Generalist | Specialist | Gain |
|---|---|---|---|
| French | xlm-t 0.772 | distilcamembert 0.949 | +0.177 |
| MSA | xlm-t 0.813 | camelbert-da 0.924 | +0.111 |
| Darija AR | xlm-t 0.723 | MARBERTv2 (fine-tuned) **0.844** | **+0.121** |
| Arabizi | xlm-t 0.762 | DarijaBERT-arabizi (fine-tuned) **0.983** | **+0.221** |

The gains are largest for the languages that had no adequate specialist before fine-tuning (Arabizi +0.221, Darija +0.121). Arabizi benefits the most because it was the worst-served language in Step 3 — xlm-t was essentially abstaining for 31% of inputs.

---

### 7.3 GPU-native models are 30–300× faster with no quality loss

All four fine-tuned models ran on a Kaggle T4 GPU during inference, producing 2–4ms per utterance. All ready-made models ran on CPU, producing 90–1006ms per utterance.

| Category | Latency range | Practical throughput |
|---|---|---|
| CPU-inferred (ready-made) | 90–1006ms | 1–11 min per 30-min programme |
| GPU-inferred (fine-tuned) | 2.5–4.4ms | 7–16 seconds per 30-min programme |

The fine-tuned models are not just more accurate — they are operationally superior in every latency dimension. A 30-minute broadcast programme (≈3600 captions at 2/second) is processed in under 16 seconds by any fine-tuned model. This enables real-time or near-real-time monitoring.

This latency advantage is partly a GPU vs CPU comparison (not entirely the model's merit), but it reflects what a production deployment would look like: fine-tuned models run on GPU infrastructure, while the ready-made models were prototyped on whatever hardware was available.

---

### 7.4 The "neutral class abstention" problem on binary test sets persists for fine-tuned models

MARBERTv2 and QARIB were fine-tuned on 3-class Darija data, which means their output head has three neurons (neg/neu/pos). When evaluated on the binary MSA test set, the neutral neuron still fires:

| Model | Test set | Neutral abstentions | Impact on recall |
|---|---|---|---|
| xlm-t | msa | 129 (12.9%) | Neg recall drops from ~0.85 to 0.750 |
| MARBERTv2 | msa | 188 (18.8%) | Neg recall drops from ~0.90 to 0.744 |
| QARIB | msa | 199 (19.9%) | Neg recall drops from ~0.90 to 0.723 |

The fine-tuned models actually abstain *more* on MSA than xlm-t does, because their neutral class was strengthened by Darija fine-tuning. This is a side effect of making the model better at Darija neutral detection — it becomes more willing to predict neutral in general, including on MSA content where neutral does not exist.

The fix is simple: fine-tune a separate model (or a separate head) on MSA data with a binary (neg/pos) output. This would be Step 4b in a more refined benchmark design.

---

### 7.5 Smaller specialist models can outperform larger ones

The benchmark repeatedly demonstrates that parameter count is not predictive of performance when the training distribution is well-matched:

| Model | Params | Best F1 | On language |
|---|---|---|---|
| distilcamembert | 68M | 0.949 | francais |
| camelbert-da | 109M | 0.924 | msa |
| QARIB | 135M | 0.827 | darija_ar |
| DarijaBERT | 147M | 0.775 | darija_ar |
| MARBERTv2 | 163M | 0.844 | darija_ar |
| DarijaBERT-arabizi | 171M | 0.983 | arabizi |
| xlm-t | 278M | 0.813 | msa |

The largest model (xlm-t, 278M) scores 0.813 on its best language. Four out of six smaller models beat it on their respective target languages. The distilcamembert result (68M → 0.949) is particularly striking: the smallest model achieves the second-highest single score in the benchmark.

---

### 7.7 LLM scaling inverts task performance: 3-class vs binary

With both Atlas-Chat sizes now evaluated, the benchmark reveals a clean inversion that depends on task structure — not just model size:

| Task | Atlas-Chat-2B | Atlas-Chat-9B | Best encoder | 9B vs 2B |
|---|---|---|---|---|
| darija_ar (3-class) | 0.727 | **0.833** | MARBERTv2 0.844 | +0.106 |
| arabizi (binary) | **0.978** | 0.754 | DarijaBERT-arabizi 0.983 | -0.224 |

**Why scaling helps on 3-class Darija:** The 9B model has a much stronger and better-calibrated sense of neutrality. Its neutral F1 jumps from 0.480 (2B) to 0.746 (9B) — nearly matching MARBERTv2's 0.751. With 9B parameters, the model has absorbed enough Arabic-language nuance to identify the absence of polar sentiment, not just its presence. This is a genuine capability that emerges with scale.

**Why scaling hurts on binary arabizi:** The same improved neutral sensitivity becomes a liability. The 9B predicts "neutre" for 42.4% of arabizi inputs (424/1000), gutting positive recall to 0.417. The 2B, with a weaker neutral prior, rounds ambiguous inputs to the nearest polar class and is mostly right (only 12 non-responses, 1.2%). The 9B is more "honest" about ambiguity — but this honesty is penalised when the evaluation set has no neutral class.

**Structural explanation:** This is a calibration-vs-task-format mismatch. The 9B is better calibrated for a 3-class world; the 2B is accidentally better calibrated for a binary world. Neither is miscalibrated in absolute terms — they just behave differently when the test set omits a class their vocabulary includes.

**Summary comparison across all LLM × language combinations:**

| Model | darija_ar | arabizi | Latency | Neutral abstention (arabizi) |
|---|---|---|---|---|
| Atlas-Chat-2B | 0.727 | **0.978** | 185 ms | 1.2% |
| Atlas-Chat-9B | **0.833** | 0.754 | 451 ms | **42.4%** |
| MARBERTv2 (fine-tuned) | **0.844** | — | 2.5 ms | — |
| DarijaBERT-arabizi (fine-tuned) | — | **0.983** | 3.9 ms | 0% |

Fine-tuned encoders remain the recommended choice for both languages. The decision rule from the plan (prefer encoder if macro-F1 ≥ 0.75 or gap < 5 pts) is satisfied in every case. However, the 9B result (0.833 on darija_ar, zero-shot) is the benchmark's most provocative finding: a 9B LLM with no task-specific training comes within 1.1 points of the best fine-tuned encoder, at 180× the latency.

---

### 7.6 DarijaBERT-arabizi: the outlier that changes the benchmark story

DarijaBERT-arabizi trained in 3.4 minutes on free Kaggle compute and achieved 0.983 macro-F1. This is an extreme result that stands apart from all other (model × language) combinations. The reasons are well-understood (perfect pre-training match + expressive binary sentiment + binary task), but it is worth noting explicitly: this is not a typical result, and one should not extrapolate from it to expect similarly extreme results in other language/model combinations.

The realistic performance range for well-matched fine-tuned models on harder tasks (3-class, morphologically complex, less expressive content) is demonstrated by MARBERTv2 on darija_ar: 0.844 — excellent, but not near-perfect. DarijaBERT-arabizi is the lucky alignment of all factors at once.

---

## 8. What the fine-tuned models actually achieved vs predictions

Section 8 of the previous version of this document made predictions for each fine-tuned model before results were available. Here is how the actual results compare.

---

### DarijaBERT on darija_ar

**Prediction:** 0.78–0.85, "realistic target". Success threshold: ≥0.75.  
**Actual:** **0.775** — just inside the predicted range, just above the threshold.

The prediction was broadly correct. The neutral recall improvement was specifically predicted ("language-specific pre-training helps the model distinguish neutral content") and confirmed (neutral F1 went from 0.577 to 0.690). The model landed at the lower end of the predicted range, which is understandable given that MARBERTv2's broader multi-dialect pre-training gave it more Moroccan coverage than DarijaBERT alone.

---

### DarijaBERT-arabizi on arabizi

**Prediction:** 0.83–0.88 ("the most anticipated model", "structural advantage from binary classification head").  
**Actual:** **0.983** — massively exceeded the prediction (+0.10 to +0.15 points above the high end).

The prediction underestimated the impact of the perfect pre-training/data match. 0.83–0.88 was a conservative estimate based on what typical fine-tuning gains look like. DarijaBERT-arabizi achieved something closer to a theoretical maximum for this task.

---

### MARBERTv2 on darija_ar

**Prediction:** 0.77–0.83, "richer Moroccan coverage than either ready-made model".  
**Actual:** **0.844** — above the predicted high end.

MARBERTv2 outperformed expectations on Darija, confirming that its 1B-word multi-dialect pre-training corpus gives it genuinely better Moroccan vocabulary coverage. The neutral F1 of 0.751 (first to break 0.75) was not explicitly predicted but follows from the general improvement in dialect coverage.

### MARBERTv2 on msa

**Prediction:** 0.89–0.92, "needs to beat camelbert-da's 0.924 — a high bar; may fall short".  
**Actual:** **0.838** — significantly below the predicted range.

The prediction assumed a dedicated MSA fine-tuning run. What actually happened was evaluation of a Darija-fine-tuned model on MSA, which introduces the 3-class head abstention problem. The 0.838 result is not a measure of MARBERTv2's MSA capability — it is a measure of what happens when a Darija-optimised model is cross-evaluated on MSA without re-training.

---

### QARIB on darija_ar

**Prediction:** 0.74–0.80, "Gulf Arabic ≈ slightly below MARBERTv2".  
**Actual:** **0.827** — above the predicted range.

QARIB performed better than expected on Darija, likely because even Gulf-focused Arabic pre-training covers enough shared morphological patterns with Moroccan Darija to provide strong fine-tuning signal.

### QARIB on msa

**Prediction:** 0.88–0.92.  
**Actual:** **0.812** — below the predicted range, for the same reason as MARBERTv2 (3-class head mismatch).

---

## 9. Figure analysis

Three figures were generated by `src/aggregate.py` and saved to `results/figs/`. This section explains what each figure shows, how to read it, and what conclusions to draw from it.

---

### 9.1 `heatmap_f1.png` — Macro-F1 by Model × Language

**What it shows:** A grid where each row is a model and each column is a language. Each filled cell shows the macro-F1 score for that (model, language) combination, with a colour scale from red (0.0) to green (1.0). Empty cells mean the model was not evaluated on that language.

**How to read the colour scale:** The scale runs from red (0.0) through yellow (~0.5) to dark green (1.0). In practice, since all scores in this benchmark are above 0.70, every cell appears somewhere in the medium-to-dark green range. The colour differences between 0.70 and 0.98 are visible but subtle — use the printed numbers, not just the colour, to compare close results.

**What the figure reveals:**

Looking at columns (languages):

- **darija_ar column:** DarijaBERT is lighter green (0.774) than MARBERTv2 (0.844) and QARIB (0.827). camelbert-da (0.701) is the lightest. The step-by-step improvement from ready-made (0.70–0.72) to fine-tuned (0.775–0.844) is visible as a colour gradient going down the column.
- **francais column:** Only two entries — distilcamembert (0.949, darkest green in the column) and xlm-t (0.772, lighter). The visual gap between them is the largest colour contrast in any single column.
- **msa column:** camelbert-da (0.924) is the darkest. MARBERTv2 (0.838) and QARIB (0.812) are both lighter, showing that the fine-tuned models do not surpass the ready-made specialist on MSA.
- **arabizi column:** Only two entries — darijabert-arabizi (0.983, darkest green in the entire figure) and xlm-t (0.762, much lighter). This contrast is the most striking in the figure.

Looking at rows (models):

- **xlm-t** is the only model with entries in all four columns — it covers the whole grid. Its colour is consistently medium green everywhere, never the best, never the worst. It is the reliable baseline.
- **camelbert-da** has two entries with very different colours: light-medium (0.701) on darija_ar, dark (0.924) on msa. This captures the "tale of two performances" described in Section 5.2.
- **darijabert-arabizi** has only one entry (arabizi) but it is the single darkest cell in the figure. One specialist language, one outstanding result.
- **marbertv2 and qarib** show symmetric two-cell patterns: both have similar colours in darija_ar and msa, with darija_ar slightly darker (better).

**Key takeaway from the heatmap:** The figure makes immediately visible that (1) no single model dominates all languages, (2) fine-tuned models are consistently better than ready-made on the languages they were trained for (darker green on darija_ar), and (3) darijabert-arabizi (0.983) and distilcamembert (0.949) on francais are the two clearest standouts in the grid.

---

### 9.2 `scatter_cost_precision.png` — Cost-Precision scatter

**What it shows:** Each model is a bubble on a 2D plot. The x-axis is the model's mean latency (ms/utterance, averaged across all its evaluated languages), plotted on a log scale. The y-axis is the model's mean macro-F1 (averaged across its evaluated languages). The bubble size represents the square root of the model's parameter count (in millions) — larger bubble = more parameters.

**How to read it:** The ideal position is **top-left**: high F1 (top) and low latency (left). The bottom-right is the worst quadrant: low F1 and slow. The log x-axis means each factor-of-10 jump in latency is the same visual distance — the gap from 3ms to 30ms looks the same as the gap from 30ms to 300ms.

**What the figure reveals:**

There is a **clear horizontal divide** between two clusters:

- **Left cluster (2–5ms):** darijabert-arabizi, marbertv2, qarib, darijabert. These are all fine-tuned models measured on GPU (Kaggle T4). They sit at the far left because GPU inference is 30–300× faster than CPU.
- **Right cluster (100–500ms):** camelbert-da (~127ms), xlm-t (~357ms average dragged right by the French latency anomaly), distilcamembert (~421ms). These are all ready-made models measured on CPU.

Within each cluster:

- In the left cluster, darijabert-arabizi sits at the top (0.983 mean F1, but note: mean is over one language only). marbertv2 (0.841 mean) and qarib (0.820 mean) are close together in the middle. darijabert (0.775 mean) is at the bottom of the left cluster.
- In the right cluster, distilcamembert has the highest mean F1 (0.949, one language) and highest latency (~421ms). camelbert-da has a moderate mean F1 (0.812, average of two very different results) and moderate latency (~127ms). xlm-t has the lowest mean F1 (0.768, average across four languages) and sits far right due to its French latency anomaly.

**Important caveat about the y-axis (mean F1):** Because models cover different numbers of languages, their mean F1 is not directly comparable. DarijaBERT-arabizi's "mean F1 of 0.983" is the F1 of its single language (arabizi), which is a binary task. MARBERTv2's "mean F1 of 0.841" is the average of its darija_ar (0.844) and msa (0.838) scores, both of which are strong results. xlm-t's "mean F1 of 0.768" is the average over four languages, dragged down by the harder darija_ar 3-class task. Models with more language coverage tend to show lower mean F1 because they include harder languages.

**The bubble size:** Bubble size represents √(params). In practice, all bubbles look similar because the parameter counts are within the same order of magnitude (68M to 278M — a factor of 4). The size differences are visible but not dramatic. The largest bubble (xlm-t, 278M) is noticeably bigger than the smallest (distilcamembert, 68M).

**Key takeaway from the scatter:** The dominant message is GPU vs CPU. The fine-tuned models did not become faster because of better architecture — they became faster because they run on GPU. If you ran the ready-made models on the same T4 GPU, their latency would also drop to 5–30ms. The scatter correctly shows the *operational* speed (how fast the model runs in the environment it will actually run in), which is the right thing to show for a deployment benchmark.

---

### 9.3 `radar_finalists.png` — Radar chart of top 5 finalists

**What it shows:** A radar (spider) chart with four axes — Precision, Integration, Cost, Coverage — and five models plotted on it. Each axis is normalised 0–1 based on the minimum and maximum scores across all seven models in the benchmark. A larger filled area = a better overall model.

**The four axes explained:**

- **Precision (right):** Normalised mean macro-F1 score (0 = worst model in the benchmark, 1 = best). DarijaBERT-arabizi scores 1.0 (highest mean F1 = 0.983), xlm-t scores 0.0 (lowest mean F1 = 0.768, though this reflects language coverage breadth as much as quality).
- **Integration (top):** Placeholder, fixed at 1.0 for all models. It represents integration complexity (API compatibility, deployment overhead) but has not been differentiated in this benchmark. Every model exposes the same HuggingFace pipeline interface, so all score the same.
- **Cost (left):** Inverted and normalised efficiency score. Higher score = lower cost (faster, less VRAM). Camelbert-da scores highest on cost (it is CPU-fast and needs no GPU VRAM, since it was measured on CPU); darijabert-arabizi scores moderately (very fast GPU inference, but GPU itself has a cost); distilcamembert scores lowest among finalists (slowest latency per its single French evaluation).
- **Coverage (bottom):** Number of languages covered divided by 4 (the total number of target languages). xlm-t (4 languages) = 1.0, but xlm-t is not in the top 5 finalists. Among the top 5: marbertv2, camelbert-da, qarib each cover 2 languages = 0.5; darijabert-arabizi and distilcamembert each cover 1 language = 0.25.

**The five models shown:**

- **darijabert-arabizi (blue):** Maximum Precision (1.0), maximum Integration (1.0), moderate Cost (~0.60), minimum Coverage (0.25). The shape is elongated toward the top-right — strong on quality, weak on breadth. Its weighted score is highest (0.845) because Precision is weighted 50%.
- **distilcamembert (orange):** High Precision (~0.84), maximum Integration (1.0), low Cost (~0.40, slow CPU), minimum Coverage (0.25). Similar shape to darijabert-arabizi but pulled inward on both Cost and Precision.
- **marbertv2 (green):** Moderate Precision (~0.34, mean F1 is not as high when normalised among all 7 models), maximum Integration, moderate Cost (~0.60), moderate Coverage (0.50). The shape is balanced but smaller than the specialists.
- **camelbert-da (red):** Low Precision (~0.21, normalised), maximum Integration, high Cost (~0.82, it's a small fast CPU model), moderate Coverage (0.50). The shape is pulled strongly toward the Cost axis — it wins on efficiency at the cost of lower precision.
- **qarib (purple):** Similar to camelbert-da but with slightly different balance — slightly higher precision, slightly lower cost score.

**What the radar shows:**

The key insight is that no model dominates all four axes. There is a fundamental tradeoff between Precision and Coverage: the highest-precision models (darijabert-arabizi, distilcamembert) only cover one language, while the broadest-coverage model (xlm-t, not shown — it fell below the top 5 cutoff) has the lowest precision. Any production system needs to decide where on this tradeoff curve to operate.

All five models max out at 1.0 on Integration because the axis is currently a placeholder. If real integration complexity were measured (deployment overhead, maintenance cost, infrastructure requirements), it would differentiate the models. A future iteration of this analysis should fill in real integration scores.

**The weighted score formula** (shown in the printed grid from aggregate.py):
```
Weighted score = 0.50 × s_precision + 0.20 × s_integration + 0.20 × s_cost + 0.10 × s_coverage
```

The weights (50% Precision, 20% Integration, 20% Cost, 10% Coverage) reflect HACA's priorities: accuracy first, then operational cost, then language breadth. Under this weighting, darijabert-arabizi wins because its exceptional precision (0.983 on arabizi) dominates the 50% precision weight, even though its coverage is minimal.

**Key takeaway from the radar:** Use this chart to understand the tradeoff space, not to identify a single winner. For a deployment that needs only Arabizi: darijabert-arabizi. For French only: distilcamembert. For a single model that handles both Darija and MSA: marbertv2. For the widest language coverage at acceptable quality: xlm-t (not in the top 5 but the only option that covers all four languages with a single model).

---

## 10. Checklist: how to analyse a new model yourself

Use this checklist every time a new result JSON lands in `results/`.

### Step 1 — Orient yourself (30 seconds)

- [ ] Open `results/summary.csv` — find the new row. Is the `macro_f1` broadly in line with expectations?
- [ ] Open the JSON. Check `classes_evaluated` — is it binary (2-class) or 3-class?
- [ ] Check `n` — should be 1000. If not, something went wrong with the test set.
- [ ] Check `n_params` — does it match the model you expected?
- [ ] Check `peak_vram_mb` — is this a GPU or CPU result? (0 = CPU; >0 = GPU)

### Step 2 — Read the per-class report (2 minutes)

- [ ] Find the class with the **lowest F1**. Note it.
- [ ] For that class: is recall low (under-predicts) or precision low (over-predicts)?
  - Low recall → the model is missing this class. What does it predict instead? (Go to matrix.)
  - Low precision → the model is over-firing on this class. What is it confusing with this class?
- [ ] Compare the class F1 scores to the xlm-t baseline on the same language. Which classes improved? Which got worse?

### Step 3 — Read the confusion matrix (3 minutes)

- [ ] Does the sum of each row equal the support for that class? If not, the model is predicting a class that's not in the matrix — note which class and how many samples.
- [ ] For each off-diagonal error cell: is the number large enough to matter? (Rough rule: > 5% of that row's support = noteworthy)
- [ ] Identify the single biggest error cell. What is the model confusing?

### Step 4 — Check latency (30 seconds)

- [ ] Compare `latency_ms_per_utt` to the xlm-t baseline on the same language.
- [ ] Is the result from GPU (peak_vram_mb > 0) or CPU? GPU results should be 2–5ms; CPU results 90–1000ms.
- [ ] Is the latency acceptable? At 1000 utterances per run and 2 runs per programme, latency above 500ms/utt becomes operationally painful.

### Step 5 — Write your interpretation (the key habit)

After reading the numbers, write 2–3 sentences in plain language answering:

> "What does this model do well, what does it do badly, and why?"

If you cannot write that sentence, you have not understood the result — go back to the confusion matrix.

Example of a good interpretation (updated with real results):
> "MARBERTv2 is the best model for Moroccan Darija (0.844 macro-F1), bringing neutral F1 above 0.75 for the first time through its broad multi-dialect Arabic pre-training. On MSA, the same model scores only 0.838 because the 3-class Darija fine-tuning head predicts 'neutral' for 18.8% of MSA inputs — a structural mismatch that would be resolved by a separate MSA-specific fine-tuning run with a binary head."

### Step 6 — Update the summary table

- [ ] Add the new result to `results/summary.csv`.
- [ ] Re-run `python src/aggregate.py` to regenerate all three figures and the weighted scoring grid.
- [ ] Update the summary table in Section 4.1 of this document.
- [ ] Update the "best model" entry in the relevant language section (Section 6).
- [ ] Note any cross-cutting pattern that was confirmed or contradicted by the new result (Section 7).

---

*This document covers Steps 3, 4, and 5. Key finding from Step 5: LLM scaling inverts task performance — Atlas-Chat-2B excels on binary arabizi (0.978, gap = 0.5 pts from DarijaBERT-arabizi) while Atlas-Chat-9B nearly matches the best fine-tuned encoder on 3-class darija_ar (0.833, gap = 1.1 pts from MARBERTv2) but collapses on arabizi (0.754) due to 42.4% neutral abstention. LLM inference is 45–180× slower than fine-tuned encoders. Fine-tuned encoders remain recommended for production. Step 6 (self-annotated domaine-réel set from real SRT files) is pending.*
