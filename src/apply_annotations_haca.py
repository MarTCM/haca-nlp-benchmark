"""
Stage 2 — Claude's hand labels for the HACA clean pool (content-valence, Rubric v3).

These labels are authored directly by Claude (no LLM API), one utterance at a time, under
report/ANNOTATION_RUBRIC_V3.md. Filled in batches; resumable.

Running this:
  * loads the clean rows of data/test_sets/haca_pool_v3.csv (deterministic order),
  * applies the LABELS dict below,
  * writes the labeled rows to data/test_sets/haca_labeled_v3.csv,
  * prints coverage + class distribution + any unlabeled remainder.

Usage:
    python src/apply_annotations_haca.py            # apply + report
    python src/apply_annotations_haca.py --batch 0  # print next 150 unlabeled (to label)
"""

import argparse
import os
import sys

import pandas as pd

POOL_CSV = "data/test_sets/haca_pool_v3.csv"
OUT_CSV  = "data/test_sets/haca_labeled_v3.csv"
BATCH_SIZE = 150
VALID = {"neg", "neu", "pos"}

# ── Claude's labels: utterance_id -> "neg" | "neu" | "pos" ───────────────────
# Filled batch by batch. Order = clean rows sorted by (file, utterance_id).
LABELS: dict[str, str] = {
    # ── Batch 1 ──────────────────────────────────────────────────────────────
    # 1.srt — garbled drama/talk-show; almost all neu (no recoverable valence),
    # except a reported child-kidnapping (0026).
    "1.srt_0009": "neu", "1.srt_0016": "neu", "1.srt_0018": "neu", "1.srt_0019": "neu",
    "1.srt_0020": "neu", "1.srt_0022": "neu", "1.srt_0023": "neu", "1.srt_0024": "neu",
    "1.srt_0026": "neg", "1.srt_0027": "neu", "1.srt_0030": "neu", "1.srt_0031": "neu",
    "1.srt_0035": "neu", "1.srt_0037": "neu", "1.srt_0038": "neu", "1.srt_0040": "neu",
    "1.srt_0042": "neu", "1.srt_0044": "neu", "1.srt_0045": "neu", "1.srt_0046": "neu",
    "1.srt_0049": "neu", "1.srt_0051": "neu", "1.srt_0052": "neu", "1.srt_0056": "neu",
    "1.srt_0057": "neu", "1.srt_0058": "neu", "1.srt_0060": "neu",
    # 10.srt — health-sector explainer. Neutral system description, a "problems"
    # block (budget/shortage/ranking/emigration/governance) = neg, and a
    # capacity-expansion reform block (more med schools, doubled intake) = pos.
    "10.srt_0004": "neg", "10.srt_0005": "neg", "10.srt_0006": "neg", "10.srt_0007": "neu",
    "10.srt_0008": "neg", "10.srt_0009": "neu", "10.srt_0011": "neu", "10.srt_0012": "neu",
    "10.srt_0015": "neu", "10.srt_0016": "neu", "10.srt_0017": "neu", "10.srt_0018": "neu",
    "10.srt_0019": "neu", "10.srt_0020": "neu", "10.srt_0021": "neu", "10.srt_0022": "neu",
    "10.srt_0023": "neu", "10.srt_0024": "neu", "10.srt_0027": "neu", "10.srt_0028": "neu",
    "10.srt_0031": "neu", "10.srt_0032": "neu", "10.srt_0033": "neu", "10.srt_0035": "neu",
    "10.srt_0036": "neu", "10.srt_0039": "neu", "10.srt_0040": "neu", "10.srt_0041": "neu",
    "10.srt_0042": "neu", "10.srt_0043": "neu", "10.srt_0045": "neu", "10.srt_0046": "neu",
    "10.srt_0047": "neu", "10.srt_0048": "neu", "10.srt_0051": "neu", "10.srt_0052": "neu",
    "10.srt_0053": "neu", "10.srt_0055": "neu", "10.srt_0056": "neu", "10.srt_0057": "neu",
    "10.srt_0058": "neu", "10.srt_0059": "neu", "10.srt_0060": "neu", "10.srt_0063": "neu",
    "10.srt_0064": "neu", "10.srt_0065": "neu", "10.srt_0066": "neu", "10.srt_0067": "neu",
    "10.srt_0068": "neu", "10.srt_0069": "neu", "10.srt_0070": "neu", "10.srt_0071": "neu",
    "10.srt_0075": "neu", "10.srt_0076": "neu", "10.srt_0077": "neu", "10.srt_0078": "neu",
    "10.srt_0079": "neu", "10.srt_0081": "neu", "10.srt_0082": "neg", "10.srt_0083": "neu",
    "10.srt_0087": "neu", "10.srt_0088": "neu", "10.srt_0089": "neu", "10.srt_0090": "neu",
    "10.srt_0091": "neu", "10.srt_0092": "neu", "10.srt_0093": "neu", "10.srt_0095": "neu",
    "10.srt_0096": "neu", "10.srt_0099": "neu", "10.srt_0100": "neu", "10.srt_0101": "neu",
    "10.srt_0102": "neu", "10.srt_0103": "neu", "10.srt_0104": "neu", "10.srt_0105": "neg",
    "10.srt_0106": "neu", "10.srt_0107": "neg", "10.srt_0108": "neu", "10.srt_0111": "neg",
    "10.srt_0112": "neg", "10.srt_0113": "neg", "10.srt_0114": "neg", "10.srt_0115": "neg",
    "10.srt_0116": "neg", "10.srt_0117": "neu", "10.srt_0118": "neu", "10.srt_0119": "neg",
    "10.srt_0123": "neg", "10.srt_0124": "neu", "10.srt_0125": "neu", "10.srt_0126": "neu",
    "10.srt_0127": "neu", "10.srt_0128": "pos", "10.srt_0129": "pos", "10.srt_0130": "pos",
    "10.srt_0131": "neu", "10.srt_0132": "neg", "10.srt_0135": "neu", "10.srt_0136": "neg",
    "10.srt_0137": "neg", "10.srt_0138": "neu", "10.srt_0139": "neu", "10.srt_0140": "neg",
    "10.srt_0141": "neg", "10.srt_0142": "neg", "10.srt_0143": "neg", "10.srt_0147": "neu",
    "10.srt_0148": "neu", "10.srt_0149": "neu", "10.srt_0150": "neg", "10.srt_0151": "neu",
    "10.srt_0152": "neu", "10.srt_0153": "neu", "10.srt_0154": "neu", "10.srt_0155": "neu",
    "10.srt_0156": "neu", "10.srt_0159": "neu", "10.srt_0160": "neg", "10.srt_0161": "neg",
    "10.srt_0162": "neg", "10.srt_0163": "neg",
    # ── Batch 2 ──────────────────────────────────────────────────────────────
    # 10.srt — corruption/ambulance/drug-price problems (neg), health-system-law
    # conclusion: liberalization grew clinics + reform fixes governance (pos).
    "10.srt_0164": "neg", "10.srt_0165": "neg", "10.srt_0166": "neg", "10.srt_0167": "neg",
    "10.srt_0168": "neg", "10.srt_0172": "neu", "10.srt_0173": "neg", "10.srt_0174": "neg",
    "10.srt_0175": "neg", "10.srt_0176": "neg", "10.srt_0177": "neg", "10.srt_0178": "neu",
    "10.srt_0179": "neg", "10.srt_0183": "neu", "10.srt_0184": "neu", "10.srt_0185": "neu",
    "10.srt_0186": "neu", "10.srt_0188": "neu", "10.srt_0189": "neu", "10.srt_0190": "neg",
    "10.srt_0191": "neg", "10.srt_0192": "neu", "10.srt_0195": "pos", "10.srt_0196": "neu",
    "10.srt_0197": "neu", "10.srt_0198": "neu", "10.srt_0199": "neu", "10.srt_0200": "neu",
    "10.srt_0201": "neu", "10.srt_0202": "neg", "10.srt_0203": "neu", "10.srt_0204": "pos",
    "10.srt_0205": "neu", "10.srt_0206": "neg", "10.srt_0207": "neu", "10.srt_0208": "neu",
    "10.srt_0209": "neu",
    # 2.srt — non-verbal-communication + medicine-storage + vitamins + smoking-law
    # + physiotherapy talk show. Advice/procedural/garbled → almost all neu.
    "2.srt_0002": "neu", "2.srt_0005": "neu", "2.srt_0006": "neu", "2.srt_0007": "neu",
    "2.srt_0008": "neu", "2.srt_0009": "neu", "2.srt_0010": "neu", "2.srt_0012": "neu",
    "2.srt_0013": "neu", "2.srt_0014": "neu", "2.srt_0017": "neu", "2.srt_0021": "neu",
    "2.srt_0022": "neu", "2.srt_0024": "neu", "2.srt_0025": "neu", "2.srt_0028": "neu",
    "2.srt_0029": "neu", "2.srt_0030": "neu", "2.srt_0033": "neu", "2.srt_0036": "neu",
    "2.srt_0037": "neu", "2.srt_0044": "neu", "2.srt_0045": "neu", "2.srt_0046": "neu",
    "2.srt_0049": "neu", "2.srt_0051": "neu", "2.srt_0052": "neu", "2.srt_0055": "neu",
    "2.srt_0056": "neu", "2.srt_0057": "neu", "2.srt_0058": "neu", "2.srt_0059": "neu",
    "2.srt_0060": "neu", "2.srt_0063": "neu", "2.srt_0064": "neu", "2.srt_0067": "neu",
    "2.srt_0070": "neu", "2.srt_0071": "neu", "2.srt_0072": "neu", "2.srt_0075": "neu",
    "2.srt_0076": "neu", "2.srt_0077": "neu", "2.srt_0078": "neu", "2.srt_0081": "neu",
    "2.srt_0083": "neu", "2.srt_0086": "neu", "2.srt_0089": "neu", "2.srt_0093": "neu",
    "2.srt_0094": "neu", "2.srt_0095": "neu", "2.srt_0096": "neu", "2.srt_0097": "neu",
    "2.srt_0100": "neu", "2.srt_0101": "neu", "2.srt_0104": "neu", "2.srt_0105": "neu",
    "2.srt_0106": "neu", "2.srt_0109": "neu", "2.srt_0112": "neg", "2.srt_0113": "neu",
    "2.srt_0120": "neu", "2.srt_0122": "neu", "2.srt_0123": "neu", "2.srt_0124": "neu",
    "2.srt_0125": "neg", "2.srt_0126": "neu", "2.srt_0128": "neu", "2.srt_0131": "neu",
    "2.srt_0132": "neu", "2.srt_0133": "neu", "2.srt_0136": "neu", "2.srt_0137": "neu",
    "2.srt_0138": "neu", "2.srt_0139": "neu", "2.srt_0142": "neu", "2.srt_0145": "neu",
    "2.srt_0147": "neu", "2.srt_0148": "neu", "2.srt_0151": "neu", "2.srt_0152": "neu",
    "2.srt_0153": "neu", "2.srt_0154": "neu", "2.srt_0155": "neu", "2.srt_0158": "neu",
    "2.srt_0159": "neu", "2.srt_0161": "neu", "2.srt_0162": "neu", "2.srt_0163": "neu",
    # 3.srt — 2025 tax-law explainer (open): pension/contribution benefits (pos),
    # middle-class burden/hogra (neg), mechanism of income tax (neu).
    "3.srt_0003": "neu", "3.srt_0004": "neu", "3.srt_0005": "pos", "3.srt_0006": "neu",
    "3.srt_0009": "neg", "3.srt_0010": "pos", "3.srt_0011": "pos", "3.srt_0012": "neu",
    "3.srt_0015": "pos", "3.srt_0016": "neu", "3.srt_0017": "neu", "3.srt_0020": "neu",
    "3.srt_0021": "neu", "3.srt_0022": "neu", "3.srt_0023": "neg", "3.srt_0026": "neg",
    "3.srt_0027": "neg", "3.srt_0028": "neu", "3.srt_0029": "neu", "3.srt_0032": "neu",
    "3.srt_0033": "neu", "3.srt_0034": "neu", "3.srt_0035": "neu", "3.srt_0038": "neu",
    "3.srt_0039": "neu",
    # ── Batch 3 ──────────────────────────────────────────────────────────────
    # 3.srt — tax mechanism (neu), fairness critique: salaried bear the burden /
    # gains are paltry / expropriation-abuse law (neg), bracket relief = pos.
    "3.srt_0040": "neu", "3.srt_0041": "neu", "3.srt_0045": "neu", "3.srt_0046": "neu",
    "3.srt_0050": "neu", "3.srt_0051": "neu", "3.srt_0052": "neu", "3.srt_0053": "neu",
    "3.srt_0056": "neu", "3.srt_0057": "neu", "3.srt_0058": "neu", "3.srt_0059": "neg",
    "3.srt_0062": "neg", "3.srt_0063": "neu", "3.srt_0064": "neu", "3.srt_0065": "neg",
    "3.srt_0068": "neu", "3.srt_0069": "neu", "3.srt_0070": "neu", "3.srt_0074": "neu",
    "3.srt_0075": "neu", "3.srt_0076": "pos", "3.srt_0077": "pos", "3.srt_0080": "neg",
    "3.srt_0081": "neg", "3.srt_0082": "neg", "3.srt_0083": "neg", "3.srt_0086": "neu",
    "3.srt_0087": "neu", "3.srt_0088": "neu", "3.srt_0089": "neu", "3.srt_0092": "neu",
    "3.srt_0093": "neu", "3.srt_0094": "neu", "3.srt_0095": "neu", "3.srt_0098": "neu",
    "3.srt_0099": "neu", "3.srt_0101": "neu", "3.srt_0102": "neu", "3.srt_0103": "neu",
    "3.srt_0104": "neu", "3.srt_0105": "neu", "3.srt_0106": "neu", "3.srt_0107": "neu",
    "3.srt_0108": "neu", "3.srt_0109": "neu", "3.srt_0110": "neu", "3.srt_0111": "neu",
    "3.srt_0112": "neu", "3.srt_0113": "neu", "3.srt_0114": "neu", "3.srt_0115": "neu",
    "3.srt_0117": "neu", "3.srt_0118": "neu", "3.srt_0119": "neu", "3.srt_0120": "neu",
    "3.srt_0121": "neu", "3.srt_0122": "neu", "3.srt_0123": "neg", "3.srt_0124": "neu",
    "3.srt_0125": "neu", "3.srt_0126": "neu", "3.srt_0127": "neu", "3.srt_0128": "neu",
    "3.srt_0129": "neu", "3.srt_0131": "neu", "3.srt_0132": "neu", "3.srt_0133": "neg",
    "3.srt_0134": "neg", "3.srt_0135": "neu", "3.srt_0138": "neg", "3.srt_0139": "neu",
    "3.srt_0140": "neu", "3.srt_0141": "neu",
    # 4.srt — public-procurement explainer. Procedural → neu; corruption framing
    # (neg); "245-billion opportunity / halal door of income" → pos.
    "4.srt_0003": "neu", "4.srt_0004": "neg", "4.srt_0005": "neu", "4.srt_0006": "neu",
    "4.srt_0007": "pos", "4.srt_0008": "neu", "4.srt_0009": "neu", "4.srt_0012": "neu",
    "4.srt_0013": "neu", "4.srt_0015": "neu", "4.srt_0017": "neu", "4.srt_0018": "pos",
    "4.srt_0021": "neg", "4.srt_0022": "neu", "4.srt_0023": "neg", "4.srt_0024": "neu",
    "4.srt_0025": "neu", "4.srt_0026": "neu", "4.srt_0027": "neu", "4.srt_0030": "neu",
    "4.srt_0031": "neu", "4.srt_0032": "neu", "4.srt_0033": "neu", "4.srt_0034": "neu",
    "4.srt_0035": "neu", "4.srt_0036": "neu", "4.srt_0039": "neu", "4.srt_0040": "neu",
    "4.srt_0041": "neu", "4.srt_0043": "neu", "4.srt_0044": "neu", "4.srt_0045": "neu",
    "4.srt_0048": "neu", "4.srt_0049": "neu", "4.srt_0050": "neu", "4.srt_0051": "neu",
    "4.srt_0052": "neu", "4.srt_0053": "neu", "4.srt_0054": "neu", "4.srt_0057": "neu",
    "4.srt_0058": "neu", "4.srt_0059": "neu", "4.srt_0060": "neu", "4.srt_0061": "neu",
    "4.srt_0062": "neu", "4.srt_0063": "neu", "4.srt_0066": "neu", "4.srt_0067": "neu",
    "4.srt_0068": "neu", "4.srt_0069": "neu", "4.srt_0070": "neu", "4.srt_0071": "neu",
    "4.srt_0072": "neu", "4.srt_0075": "neu", "4.srt_0076": "neu", "4.srt_0077": "neu",
    "4.srt_0078": "neu", "4.srt_0079": "neu", "4.srt_0080": "neu", "4.srt_0081": "neu",
    "4.srt_0084": "neu", "4.srt_0085": "neu", "4.srt_0086": "neu", "4.srt_0087": "neu",
    "4.srt_0088": "neu", "4.srt_0089": "neu", "4.srt_0090": "neu", "4.srt_0093": "neu",
    "4.srt_0094": "neu", "4.srt_0095": "neu", "4.srt_0096": "neu", "4.srt_0097": "neu",
    "4.srt_0098": "neu", "4.srt_0099": "neu", "4.srt_0102": "neu", "4.srt_0103": "neu",
    # ── Batch 4 ──────────────────────────────────────────────────────────────
    # 4.srt — rest of procurement how-to + the "be the insider via good service"
    # relational close. All procedural/relational → neu.
    "4.srt_0104": "neu", "4.srt_0105": "neu", "4.srt_0106": "neu", "4.srt_0107": "neu",
    "4.srt_0108": "neu", "4.srt_0111": "neu", "4.srt_0112": "neu", "4.srt_0113": "neu",
    "4.srt_0115": "neu", "4.srt_0116": "neu", "4.srt_0117": "neu", "4.srt_0120": "neu",
    "4.srt_0121": "neu", "4.srt_0122": "neu", "4.srt_0123": "neu", "4.srt_0124": "neu",
    "4.srt_0125": "neu", "4.srt_0129": "neu", "4.srt_0130": "neu", "4.srt_0131": "neu",
    "4.srt_0135": "neu", "4.srt_0138": "neu", "4.srt_0139": "neu", "4.srt_0140": "neu",
    "4.srt_0141": "neu", "4.srt_0142": "neu", "4.srt_0143": "neu", "4.srt_0144": "neu",
    "4.srt_0147": "neu", "4.srt_0148": "neu", "4.srt_0149": "neu", "4.srt_0150": "neu",
    "4.srt_0151": "neu", "4.srt_0152": "neu", "4.srt_0155": "neu",
    # 5.srt — stock-market explainer. Mechanism/history → neu; "multiply your
    # money / invest comfortably with little" → pos; crash/loss framing → neg.
    "5.srt_0005": "neg", "5.srt_0006": "pos", "5.srt_0010": "neu", "5.srt_0011": "neu",
    "5.srt_0012": "neu", "5.srt_0016": "neu", "5.srt_0017": "neu", "5.srt_0021": "neu",
    "5.srt_0022": "neu", "5.srt_0024": "neu", "5.srt_0027": "neu", "5.srt_0028": "neu",
    "5.srt_0029": "neu", "5.srt_0030": "neu", "5.srt_0033": "neu", "5.srt_0034": "neu",
    "5.srt_0035": "neu", "5.srt_0036": "neu", "5.srt_0040": "neu", "5.srt_0041": "neu",
    "5.srt_0042": "neu", "5.srt_0045": "neu", "5.srt_0046": "neu", "5.srt_0047": "neu",
    "5.srt_0048": "neu", "5.srt_0051": "neu", "5.srt_0052": "neu", "5.srt_0054": "neu",
    "5.srt_0057": "neu", "5.srt_0058": "neu", "5.srt_0059": "neu", "5.srt_0060": "neu",
    "5.srt_0063": "neu", "5.srt_0065": "neu", "5.srt_0066": "neu", "5.srt_0069": "neu",
    "5.srt_0070": "neu", "5.srt_0071": "neu", "5.srt_0075": "neu", "5.srt_0076": "neu",
    "5.srt_0077": "neu", "5.srt_0078": "neu", "5.srt_0081": "neu", "5.srt_0082": "neu",
    "5.srt_0083": "neu", "5.srt_0084": "neu", "5.srt_0087": "neu", "5.srt_0088": "neu",
    "5.srt_0089": "neu", "5.srt_0092": "neu", "5.srt_0093": "neu", "5.srt_0095": "neu",
    "5.srt_0098": "neu", "5.srt_0099": "neu", "5.srt_0100": "neu", "5.srt_0101": "neu",
    "5.srt_0102": "neu", "5.srt_0103": "neu", "5.srt_0104": "neu", "5.srt_0105": "neu",
    "5.srt_0106": "neu", "5.srt_0107": "neu", "5.srt_0108": "neu", "5.srt_0109": "neu",
    "5.srt_0110": "neu", "5.srt_0111": "neu", "5.srt_0112": "neu", "5.srt_0113": "neu",
    "5.srt_0115": "neu", "5.srt_0116": "neu", "5.srt_0117": "neu", "5.srt_0118": "neu",
    "5.srt_0119": "neu", "5.srt_0120": "neu", "5.srt_0121": "neu", "5.srt_0122": "neu",
    "5.srt_0123": "neu", "5.srt_0125": "neu", "5.srt_0126": "neu", "5.srt_0127": "neu",
    "5.srt_0128": "neu", "5.srt_0129": "neu", "5.srt_0130": "neu", "5.srt_0131": "neu",
    "5.srt_0132": "neu", "5.srt_0133": "neu", "5.srt_0134": "neu", "5.srt_0135": "neu",
    "5.srt_0136": "neu", "5.srt_0137": "neu", "5.srt_0138": "neu", "5.srt_0139": "pos",
    "5.srt_0140": "neu", "5.srt_0141": "neu", "5.srt_0143": "neu",
    # 6.srt — Sufi/religious + "wisdom" (الحكمة) programme. Spiritual recitation /
    # definitional / moral exhortation → all neu (rubric: religious → neu).
    "6.srt_0016": "neu", "6.srt_0021": "neu", "6.srt_0023": "neu", "6.srt_0025": "neu",
    "6.srt_0026": "neu", "6.srt_0035": "neu", "6.srt_0038": "neu", "6.srt_0040": "neu",
    "6.srt_0041": "neu", "6.srt_0042": "neu", "6.srt_0045": "neu", "6.srt_0046": "neu",
    "6.srt_0048": "neu", "6.srt_0052": "neu", "6.srt_0054": "neu", "6.srt_0057": "neu",
    "6.srt_0058": "neu", "6.srt_0060": "neu", "6.srt_0061": "neu", "6.srt_0062": "neu",
    # ── Batch 5 ──────────────────────────────────────────────────────────────
    # 6.srt — rest of wisdom/freedom sermon + a hospital TV-drama portion → all neu.
    "6.srt_0064": "neu", "6.srt_0068": "neu", "6.srt_0073": "neu", "6.srt_0075": "neu",
    "6.srt_0076": "neu", "6.srt_0077": "neu", "6.srt_0079": "neu", "6.srt_0081": "neu",
    "6.srt_0083": "neu", "6.srt_0084": "neu", "6.srt_0086": "neu", "6.srt_0088": "neu",
    "6.srt_0089": "neu", "6.srt_0092": "neu", "6.srt_0094": "neu", "6.srt_0095": "neu",
    "6.srt_0097": "neu", "6.srt_0101": "neu", "6.srt_0102": "neu", "6.srt_0103": "neu",
    "6.srt_0105": "neu", "6.srt_0106": "neu", "6.srt_0107": "neu",
    # 7.srt — Sahara history (historian Nabil Mouline). Historical narration = neu;
    # colonial loss/partition/war setbacks = neg; liberation, Green March,
    # territorial development, 2017 AU return + 2020/2025 sovereignty recognitions = pos.
    "7.srt_0003": "neu", "7.srt_0004": "neu", "7.srt_0005": "neu", "7.srt_0006": "neu",
    "7.srt_0007": "neu", "7.srt_0008": "neu", "7.srt_0009": "neu", "7.srt_0012": "neu",
    "7.srt_0013": "neu", "7.srt_0014": "neu", "7.srt_0015": "neu", "7.srt_0016": "neu",
    "7.srt_0017": "neu", "7.srt_0018": "neu", "7.srt_0021": "neu", "7.srt_0022": "neu",
    "7.srt_0023": "neu", "7.srt_0024": "neu", "7.srt_0025": "neu", "7.srt_0026": "neu",
    "7.srt_0027": "neu", "7.srt_0030": "neu", "7.srt_0031": "neu", "7.srt_0032": "neu",
    "7.srt_0033": "neu", "7.srt_0034": "neu", "7.srt_0036": "neu", "7.srt_0039": "pos",
    "7.srt_0041": "neu", "7.srt_0042": "neu", "7.srt_0043": "neu", "7.srt_0044": "neu",
    "7.srt_0045": "neu", "7.srt_0048": "neg", "7.srt_0049": "neg", "7.srt_0050": "neu",
    "7.srt_0051": "neu", "7.srt_0052": "neu", "7.srt_0053": "neg", "7.srt_0057": "neu",
    "7.srt_0060": "neg", "7.srt_0061": "neg", "7.srt_0062": "neu", "7.srt_0066": "neu",
    "7.srt_0067": "neu", "7.srt_0068": "neg", "7.srt_0069": "neu", "7.srt_0070": "neu",
    "7.srt_0071": "neu", "7.srt_0072": "neu", "7.srt_0075": "neu", "7.srt_0076": "neu",
    "7.srt_0077": "neg", "7.srt_0078": "neu", "7.srt_0079": "neu", "7.srt_0080": "neu",
    "7.srt_0081": "neu", "7.srt_0084": "pos", "7.srt_0085": "pos", "7.srt_0086": "pos",
    "7.srt_0087": "neu", "7.srt_0088": "neu", "7.srt_0089": "neu", "7.srt_0090": "neu",
    "7.srt_0093": "neu", "7.srt_0094": "neg", "7.srt_0095": "neg", "7.srt_0096": "neg",
    "7.srt_0097": "neg", "7.srt_0098": "neu", "7.srt_0099": "neu", "7.srt_0103": "neu",
    "7.srt_0104": "neu", "7.srt_0105": "neu", "7.srt_0106": "neu", "7.srt_0107": "neu",
    "7.srt_0108": "neu", "7.srt_0111": "pos", "7.srt_0112": "neu", "7.srt_0113": "neg",
    "7.srt_0114": "neg", "7.srt_0115": "neu", "7.srt_0117": "neu", "7.srt_0120": "neu",
    "7.srt_0121": "neu", "7.srt_0122": "pos", "7.srt_0123": "neu", "7.srt_0124": "pos",
    "7.srt_0125": "pos", "7.srt_0126": "pos", "7.srt_0129": "neu", "7.srt_0131": "neu",
    "7.srt_0132": "neu", "7.srt_0133": "neu", "7.srt_0135": "pos", "7.srt_0138": "pos",
    "7.srt_0139": "pos", "7.srt_0140": "pos", "7.srt_0141": "pos", "7.srt_0142": "neu",
    "7.srt_0143": "neu", "7.srt_0144": "neu", "7.srt_0147": "neu", "7.srt_0148": "neu",
    "7.srt_0149": "neu", "7.srt_0150": "neu", "7.srt_0151": "neu", "7.srt_0152": "neu",
    "7.srt_0153": "neu", "7.srt_0154": "neu", "7.srt_0155": "neu", "7.srt_0157": "neu",
    "7.srt_0160": "neu", "7.srt_0161": "neu",
    # 7769.srt — "نقطة إلى السطر" political debate (PJD on a censure motion).
    # Opposition accusations: suspicious withdrawal / buying-and-selling = neg.
    "7769.srt_0004": "neu", "7769.srt_0011": "neu", "7769.srt_0012": "neg",
    "7769.srt_0013": "neg", "7769.srt_0014": "neg", "7769.srt_0015": "neg",
    "7769.srt_0016": "neg", "7769.srt_0017": "neg", "7769.srt_0018": "neu",
    "7769.srt_0020": "neg", "7769.srt_0021": "neg", "7769.srt_0022": "neu",
    "7769.srt_0023": "neu",
    # ── Batch 6 ──────────────────────────────────────────────────────────────
    # 7769.srt — rest of the PJD debate. Opposition attacks: import scandal,
    # broken promises, unemployment, anti-corruption inaction, PM conflict of
    # interest (water desalination) = neg. The party-congress self-praise
    # ("brilliant success", "a political wedding for the nation") = pos.
    "7769.srt_0024": "neg", "7769.srt_0025": "neg", "7769.srt_0026": "neu",
    "7769.srt_0027": "neu", "7769.srt_0030": "neg", "7769.srt_0032": "neg",
    "7769.srt_0033": "neg", "7769.srt_0034": "neu", "7769.srt_0035": "neu",
    "7769.srt_0036": "neg", "7769.srt_0037": "neg", "7769.srt_0038": "neg",
    "7769.srt_0039": "neu", "7769.srt_0040": "neg", "7769.srt_0041": "neu",
    "7769.srt_0043": "neu", "7769.srt_0044": "neg", "7769.srt_0046": "neu",
    "7769.srt_0048": "neg", "7769.srt_0049": "neu", "7769.srt_0050": "neu",
    "7769.srt_0051": "neu", "7769.srt_0052": "neu", "7769.srt_0053": "neu",
    "7769.srt_0054": "neu", "7769.srt_0055": "neg", "7769.srt_0062": "neg",
    "7769.srt_0063": "neu", "7769.srt_0064": "neg", "7769.srt_0065": "neg",
    "7769.srt_0066": "neu", "7769.srt_0069": "neg", "7769.srt_0070": "neu",
    "7769.srt_0071": "neg", "7769.srt_0072": "neg", "7769.srt_0073": "neu",
    "7769.srt_0076": "neg", "7769.srt_0077": "neg", "7769.srt_0080": "neg",
    "7769.srt_0081": "neg", "7769.srt_0082": "neg", "7769.srt_0085": "neg",
    "7769.srt_0086": "neg", "7769.srt_0091": "neu", "7769.srt_0092": "neu",
    "7769.srt_0094": "neu", "7769.srt_0095": "neg", "7769.srt_0096": "neg",
    "7769.srt_0098": "neu", "7769.srt_0099": "neg", "7769.srt_0100": "neu",
    "7769.srt_0105": "neu", "7769.srt_0106": "neg", "7769.srt_0107": "neu",
    "7769.srt_0108": "neg", "7769.srt_0109": "neg", "7769.srt_0110": "neg",
    "7769.srt_0115": "neu", "7769.srt_0116": "neu", "7769.srt_0117": "neg",
    "7769.srt_0118": "neu", "7769.srt_0122": "neg", "7769.srt_0123": "neg",
    "7769.srt_0124": "neg", "7769.srt_0125": "neg", "7769.srt_0126": "neu",
    "7769.srt_0127": "neg", "7769.srt_0128": "neu", "7769.srt_0129": "neg",
    "7769.srt_0130": "neg", "7769.srt_0131": "neg", "7769.srt_0132": "neg",
    "7769.srt_0133": "neu", "7769.srt_0134": "neu", "7769.srt_0135": "neu",
    "7769.srt_0136": "neu", "7769.srt_0137": "neu", "7769.srt_0138": "neu",
    "7769.srt_0139": "neg", "7769.srt_0141": "neu", "7769.srt_0145": "neg",
    "7769.srt_0146": "neu", "7769.srt_0147": "neu", "7769.srt_0148": "neu",
    "7769.srt_0149": "neu", "7769.srt_0150": "neu", "7769.srt_0151": "neg",
    "7769.srt_0152": "neg", "7769.srt_0153": "neg", "7769.srt_0154": "neg",
    "7769.srt_0155": "neu", "7769.srt_0156": "neu", "7769.srt_0157": "neu",
    "7769.srt_0158": "neu", "7769.srt_0160": "neu", "7769.srt_0161": "neu",
    "7769.srt_0162": "neu", "7769.srt_0163": "neu", "7769.srt_0164": "pos",
    "7769.srt_0165": "pos", "7769.srt_0166": "neu", "7769.srt_0167": "neu",
    "7769.srt_0168": "neu", "7769.srt_0171": "neu", "7769.srt_0172": "neu",
    "7769.srt_0176": "neu", "7769.srt_0177": "neu", "7769.srt_0178": "neu",
    "7769.srt_0179": "neu", "7769.srt_0180": "neu", "7769.srt_0181": "neu",
    "7769.srt_0182": "neu", "7769.srt_0183": "pos", "7769.srt_0184": "pos",
    "7769.srt_0185": "pos", "7769.srt_0186": "neu", "7769.srt_0187": "neu",
    "7769.srt_0188": "neu", "7769.srt_0189": "neu", "7769.srt_0190": "neu",
    "7769.srt_0192": "neu", "7769.srt_0196": "neu", "7769.srt_0197": "neu",
    "7769.srt_0198": "neu", "7769.srt_0199": "neu", "7769.srt_0200": "neu",
    "7769.srt_0202": "neu", "7769.srt_0203": "neu", "7769.srt_0204": "neu",
    "7769.srt_0205": "neu", "7769.srt_0206": "neu", "7769.srt_0207": "neu",
    "7769.srt_0208": "neg", "7769.srt_0209": "neu", "7769.srt_0210": "neu",
    # 7770.srt — USFP debate (open). Grievances list (strikes, cost of living,
    # cattle import) = neg; the rest procedural narration = neu.
    "7770.srt_0003": "neu", "7770.srt_0004": "neu", "7770.srt_0009": "neu",
    "7770.srt_0010": "neu", "7770.srt_0012": "neu", "7770.srt_0013": "neg",
    "7770.srt_0014": "neu", "7770.srt_0017": "neu", "7770.srt_0019": "neu",
    "7770.srt_0020": "neu", "7770.srt_0021": "neu", "7770.srt_0022": "neu",
    "7770.srt_0024": "neu", "7770.srt_0025": "neu", "7770.srt_0026": "neu",
    # ── Batch 7 ──────────────────────────────────────────────────────────────
    # 7770.srt — rest of the USFP debate. Accusations (lies, "deal", PJD
    # obstruction, govt "taghawul"/domination, price/education failures) = neg;
    # World Cup hosting + national unity = pos; the rest = neu.
    "7770.srt_0028": "neu", "7770.srt_0029": "neu", "7770.srt_0030": "neu",
    "7770.srt_0031": "neu", "7770.srt_0035": "neu", "7770.srt_0036": "neu",
    "7770.srt_0038": "neu", "7770.srt_0039": "neu", "7770.srt_0040": "neu",
    "7770.srt_0043": "neu", "7770.srt_0044": "neu", "7770.srt_0045": "neu",
    "7770.srt_0049": "neu", "7770.srt_0050": "neu", "7770.srt_0052": "neu",
    "7770.srt_0053": "neu", "7770.srt_0056": "neu", "7770.srt_0060": "neu",
    "7770.srt_0061": "neu", "7770.srt_0062": "neg", "7770.srt_0063": "neu",
    "7770.srt_0064": "neu", "7770.srt_0065": "neg", "7770.srt_0066": "neu",
    "7770.srt_0068": "neu", "7770.srt_0069": "neu", "7770.srt_0070": "neu",
    "7770.srt_0071": "neu", "7770.srt_0073": "neg", "7770.srt_0074": "neu",
    "7770.srt_0075": "neu", "7770.srt_0076": "neu", "7770.srt_0077": "neg",
    "7770.srt_0078": "neg", "7770.srt_0079": "neu", "7770.srt_0080": "neg",
    "7770.srt_0084": "neu", "7770.srt_0085": "neg", "7770.srt_0087": "neu",
    "7770.srt_0088": "neu", "7770.srt_0092": "neu", "7770.srt_0093": "neu",
    "7770.srt_0094": "neu", "7770.srt_0095": "neu", "7770.srt_0096": "neu",
    "7770.srt_0097": "neu", "7770.srt_0098": "neu", "7770.srt_0099": "neu",
    "7770.srt_0100": "neu", "7770.srt_0101": "neu", "7770.srt_0102": "neu",
    "7770.srt_0103": "neg", "7770.srt_0104": "neu", "7770.srt_0105": "neu",
    "7770.srt_0106": "neu", "7770.srt_0107": "neu", "7770.srt_0108": "neu",
    "7770.srt_0109": "neu", "7770.srt_0111": "neu", "7770.srt_0112": "neu",
    "7770.srt_0116": "neg", "7770.srt_0117": "neg", "7770.srt_0118": "neg",
    "7770.srt_0119": "neu", "7770.srt_0120": "neu", "7770.srt_0121": "neu",
    "7770.srt_0122": "neg", "7770.srt_0125": "neu", "7770.srt_0126": "neu",
    "7770.srt_0127": "neu", "7770.srt_0129": "neu", "7770.srt_0130": "neu",
    "7770.srt_0134": "neu", "7770.srt_0135": "neu", "7770.srt_0136": "neu",
    "7770.srt_0137": "neu", "7770.srt_0139": "neu", "7770.srt_0140": "neu",
    "7770.srt_0141": "neu", "7770.srt_0142": "neu", "7770.srt_0144": "neu",
    "7770.srt_0145": "neu", "7770.srt_0147": "neu", "7770.srt_0148": "neu",
    "7770.srt_0149": "neu", "7770.srt_0153": "neu", "7770.srt_0154": "neu",
    "7770.srt_0155": "neu", "7770.srt_0156": "neu", "7770.srt_0157": "neu",
    "7770.srt_0159": "neu", "7770.srt_0163": "neu", "7770.srt_0164": "neg",
    "7770.srt_0165": "neg", "7770.srt_0166": "neg", "7770.srt_0171": "neg",
    "7770.srt_0172": "neu", "7770.srt_0174": "neg", "7770.srt_0175": "neu",
    "7770.srt_0176": "neu", "7770.srt_0177": "neu", "7770.srt_0178": "neu",
    "7770.srt_0179": "neg", "7770.srt_0181": "neg", "7770.srt_0182": "neu",
    "7770.srt_0183": "neu", "7770.srt_0184": "neu", "7770.srt_0185": "neg",
    "7770.srt_0190": "neu", "7770.srt_0191": "neu", "7770.srt_0192": "neu",
    "7770.srt_0195": "neg", "7770.srt_0196": "neg", "7770.srt_0197": "neu",
    "7770.srt_0198": "neu", "7770.srt_0199": "pos", "7770.srt_0201": "neu",
    "7770.srt_0202": "neu", "7770.srt_0203": "neu", "7770.srt_0204": "neu",
    "7770.srt_0205": "neu", "7770.srt_0206": "neu", "7770.srt_0207": "neu",
    "7770.srt_0208": "neu", "7770.srt_0209": "neu", "7770.srt_0210": "neu",
    # 8.srt — corruption documentary. Corruption harms / backwardness /
    # wasted investment / low CPI score = neg; definitions/setup = neu.
    "8.srt_0003": "neg", "8.srt_0005": "neg", "8.srt_0006": "neg", "8.srt_0010": "neg",
    "8.srt_0012": "neu", "8.srt_0015": "neu", "8.srt_0016": "neu", "8.srt_0017": "neg",
    "8.srt_0018": "neu", "8.srt_0021": "neg", "8.srt_0022": "neg", "8.srt_0023": "neg",
    "8.srt_0024": "neu", "8.srt_0027": "neg", "8.srt_0028": "neg", "8.srt_0029": "neg",
    "8.srt_0030": "neg", "8.srt_0033": "neu", "8.srt_0034": "neu", "8.srt_0035": "neg",
    "8.srt_0036": "neu", "8.srt_0039": "neg", "8.srt_0040": "neg", "8.srt_0041": "neg",
    # ── Batch 8 (final) ──────────────────────────────────────────────────────
    # 8.srt — rest of corruption documentary. Legal definitions/procedure = neu;
    # declining CPI score, ineffective declaration, legislative voids, shelved
    # laws, commission that never met, brain-drain, everyday bribery = neg.
    "8.srt_0042": "neu", "8.srt_0045": "neu", "8.srt_0046": "neu", "8.srt_0048": "neu",
    "8.srt_0051": "neg", "8.srt_0057": "neu", "8.srt_0058": "neu", "8.srt_0059": "neu",
    "8.srt_0060": "neg", "8.srt_0063": "neu", "8.srt_0064": "neg", "8.srt_0065": "neu",
    "8.srt_0066": "neu", "8.srt_0069": "neg", "8.srt_0070": "neg", "8.srt_0071": "neu",
    "8.srt_0075": "neu", "8.srt_0076": "neu", "8.srt_0077": "neu", "8.srt_0081": "neu",
    "8.srt_0082": "neu", "8.srt_0083": "neu", "8.srt_0084": "neu", "8.srt_0088": "neg",
    "8.srt_0089": "neu", "8.srt_0090": "neu", "8.srt_0093": "neu", "8.srt_0094": "neu",
    "8.srt_0095": "neu", "8.srt_0097": "neu", "8.srt_0098": "neu", "8.srt_0099": "neu",
    "8.srt_0100": "neg", "8.srt_0101": "neu", "8.srt_0102": "neu", "8.srt_0103": "neu",
    "8.srt_0104": "neu", "8.srt_0105": "neu", "8.srt_0107": "neg", "8.srt_0108": "neg",
    "8.srt_0109": "neg", "8.srt_0111": "neu", "8.srt_0112": "neg", "8.srt_0113": "neu",
    "8.srt_0115": "neg", "8.srt_0116": "neg", "8.srt_0117": "neg", "8.srt_0119": "neu",
    "8.srt_0120": "neu", "8.srt_0121": "neu", "8.srt_0122": "neu", "8.srt_0123": "neu",
    "8.srt_0124": "neu", "8.srt_0125": "neu",
    # 9.srt — ancient Maghreb history (Mouline). Narration of dynasties = neu;
    # hostile Algeria rhetoric / decline + foreign invasion / wounded dignity = neg.
    "9.srt_0003": "neu", "9.srt_0005": "neu", "9.srt_0006": "neu", "9.srt_0007": "neu",
    "9.srt_0008": "neg", "9.srt_0009": "neu", "9.srt_0012": "neu", "9.srt_0013": "neu",
    "9.srt_0014": "neu", "9.srt_0016": "neu", "9.srt_0017": "neu", "9.srt_0018": "neu",
    "9.srt_0021": "neu", "9.srt_0022": "neu", "9.srt_0023": "neu", "9.srt_0024": "neu",
    "9.srt_0025": "neu", "9.srt_0026": "neu", "9.srt_0030": "neu", "9.srt_0031": "neu",
    "9.srt_0032": "neu", "9.srt_0035": "neu", "9.srt_0036": "neu", "9.srt_0039": "neu",
    "9.srt_0040": "neu", "9.srt_0041": "neu", "9.srt_0042": "neu", "9.srt_0043": "neu",
    "9.srt_0044": "neu", "9.srt_0045": "neu", "9.srt_0048": "neu", "9.srt_0049": "neu",
    "9.srt_0050": "neu", "9.srt_0051": "neu", "9.srt_0052": "neu", "9.srt_0053": "neu",
    "9.srt_0054": "neu", "9.srt_0058": "neu", "9.srt_0060": "neu", "9.srt_0061": "neu",
    "9.srt_0062": "neu", "9.srt_0063": "neu", "9.srt_0066": "neu", "9.srt_0067": "neu",
    "9.srt_0068": "neu", "9.srt_0071": "neu", "9.srt_0072": "neu", "9.srt_0075": "neu",
    "9.srt_0076": "neg", "9.srt_0077": "neg", "9.srt_0078": "neu", "9.srt_0079": "neu",
    "9.srt_0080": "neu", "9.srt_0081": "neu", "9.srt_0084": "neu", "9.srt_0085": "neu",
    "9.srt_0086": "neu", "9.srt_0087": "neu", "9.srt_0088": "neu", "9.srt_0090": "neu",
    "9.srt_0093": "neu", "9.srt_0094": "neu", "9.srt_0096": "neu", "9.srt_0098": "neg",
    "9.srt_0099": "neu", "9.srt_0102": "neu", "9.srt_0104": "neu", "9.srt_0105": "neu",
    "9.srt_0106": "neu", "9.srt_0108": "neu", "9.srt_0111": "neu", "9.srt_0112": "neu",
    "9.srt_0113": "neu", "9.srt_0115": "neu", "9.srt_0116": "neu", "9.srt_0120": "neu",
    "9.srt_0121": "neu", "9.srt_0123": "neu", "9.srt_0124": "neu", "9.srt_0126": "neu",
    "9.srt_0129": "neu", "9.srt_0130": "neu", "9.srt_0132": "neu", "9.srt_0133": "neu",
    "9.srt_0134": "neu", "9.srt_0135": "neu", "9.srt_0138": "neu", "9.srt_0139": "neu",
    "9.srt_0140": "neu", "9.srt_0141": "neu", "9.srt_0142": "neu", "9.srt_0143": "neu",
    "9.srt_0145": "neu", "9.srt_0146": "neu", "9.srt_0148": "neu", "9.srt_0149": "neu",
}


def clean_pool() -> pd.DataFrame:
    df = pd.read_csv(POOL_CSV)
    df = df[df["quality"] == "clean"].copy()
    # deterministic order for resumable batching
    df = df.sort_values(["file", "utterance_id"]).reset_index(drop=True)
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", type=int, default=None,
                    help="Print the Nth batch of still-unlabeled utterances (for labeling).")
    ap.add_argument("--size", type=int, default=BATCH_SIZE)
    args = ap.parse_args()

    df = clean_pool()
    total = len(df)

    bad = {k: v for k, v in LABELS.items() if v not in VALID}
    if bad:
        print(f"!! invalid labels: {bad}", file=sys.stderr)

    if args.batch is not None:
        # Print the next `size` utterances that have no label yet, in order.
        unl = df[~df["utterance_id"].isin(LABELS)].reset_index(drop=True)
        start = args.batch * args.size
        chunk = unl.iloc[start:start + args.size]
        print(f"# unlabeled remaining: {len(unl)}  | showing rows {start}..{start+len(chunk)-1}")
        for _, r in chunk.iterrows():
            print(f"{r['utterance_id']}\t{r['text']}")
        return

    # Apply + report
    labeled = df[df["utterance_id"].isin(LABELS)].copy()
    labeled["label"] = labeled["utterance_id"].map(LABELS)
    labeled["label_source"] = "claude-manual"

    cols = ["utterance_id", "file", "fmt", "detected_lang", "quality", "text",
            "label", "label_source"]
    labeled[cols].to_csv(OUT_CSV, index=False)

    print(f"Clean pool      : {total}")
    print(f"Labeled         : {len(labeled)}  ({len(labeled)/total*100:.0f}%)")
    print(f"Remaining       : {total - len(labeled)}")
    print(f"Distribution    : {dict(labeled['label'].value_counts())}")
    print(f"Wrote           : {OUT_CSV}")


if __name__ == "__main__":
    main()
