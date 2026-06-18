"""
Calibrate per-class decision thresholds for MARBERTv2 on the domaine-réel test set.

Idea: instead of argmax(scores), predict argmax(score - threshold).
Lowering T_neg below 0.5 recovers negatives that the model assigns a moderate
neg score to — which is the main failure mode on broadcast content.

5-fold cross-validation avoids optimising and evaluating on the same 194 utterances.
The "global" thresholds (fit on the full dataset) are saved for deployment.

Usage:
    python src/calibrate_thresholds.py
    python src/calibrate_thresholds.py --model marbertv2   # default
    python src/calibrate_thresholds.py --model darijabert
"""

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd
import torch
from itertools import product
from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import StratifiedKFold
from transformers import pipeline, AutoTokenizer

sys.path.insert(0, os.path.dirname(__file__))
from label_maps import FINETUNED_MAP, apply_map

DEFAULT_TEST_CSV = "data/test_sets/domaine_reel_v2.csv"   # canonical gold (content-valence)
DEVICE   = 0 if torch.cuda.is_available() else -1
BATCH    = 16
N_FOLDS  = 5
CLASSES  = ["neg", "neu", "pos"]

MODEL_REGISTRY = {
    "marbertv2":      ("checkpoints/marbertv2/checkpoint-1311",  "UBC-NLP/MARBERTv2"),
    "darijabert":     ("checkpoints/darijabert/checkpoint-1311", "SI2M-Lab/DarijaBERT"),
    "qarib":          ("checkpoints/qarib/checkpoint-1311",      "qarib/bert-base-qarib"),
    # Stage 5 (v3) — best model saved directly to checkpoints/<key>
    "marbertv2-haca":  ("checkpoints/marbertv2-haca",  "UBC-NLP/MARBERTv2"),
    "darijabert-haca": ("checkpoints/darijabert-haca", "SI2M-Lab/DarijaBERT"),
    "marbertv2-haca-only":     ("checkpoints/marbertv2-haca-only",     "UBC-NLP/MARBERTv2"),
    "marbertv2-haca-only-hub": ("checkpoints/marbertv2-haca-only-hub", "UBC-NLP/MARBERTv2"),
}

# Grid: T_neg and T_pos from 0.05 to 0.60 — 12×12 = 144 combinations
GRID = np.round(np.arange(0.05, 0.61, 0.05), 2)


def collect_scores(pipe, texts: list) -> list:
    """Return list of {neg: float, neu: float, pos: float} for each text."""
    raw = pipe(texts, batch_size=BATCH, truncation=True, max_length=512)
    out = []
    for item in raw:
        scores = item if isinstance(item, list) else [item]
        out.append({apply_map(s["label"], FINETUNED_MAP): s["score"] for s in scores})
    return out


def shifted_argmax(scores_list: list, t_neg: float, t_pos: float) -> list:
    """
    Predict argmax of (score - threshold).
    T_neu is fixed at 0.5 (we don't shift the majority class).
    """
    preds = []
    for s in scores_list:
        shifted = {
            "neg": s.get("neg", 0.0) - t_neg,
            "neu": s.get("neu", 0.0) - 0.5,
            "pos": s.get("pos", 0.0) - t_pos,
        }
        preds.append(max(shifted, key=shifted.get))
    return preds


def grid_search(scores_list: list, y_true: list) -> tuple:
    """Return (t_neg, t_pos, macro_f1) maximising macro-F1 on the given data."""
    best_f1, best_tn, best_tp = 0.0, 0.5, 0.5
    for t_neg, t_pos in product(GRID, GRID):
        preds = shifted_argmax(scores_list, t_neg, t_pos)
        f1 = f1_score(y_true, preds, labels=CLASSES, average="macro", zero_division=0)
        if f1 > best_f1:
            best_f1, best_tn, best_tp = f1, float(t_neg), float(t_pos)
    return best_tn, best_tp, best_f1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="marbertv2", choices=list(MODEL_REGISTRY))
    parser.add_argument("--test-csv", default=DEFAULT_TEST_CSV,
                        help="Gold test set to calibrate/evaluate on (default: domaine_reel_v2).")
    args = parser.parse_args()

    model_path, tok_src = MODEL_REGISTRY[args.model]
    TEST_CSV = args.test_csv

    df     = pd.read_csv(TEST_CSV)
    texts  = df["text"].tolist()
    y_true = df["label"].tolist()

    print(f"Model : {args.model}")
    print(f"Data  : {TEST_CSV}  (n={len(texts)}, neg={y_true.count('neg')}, "
          f"neu={y_true.count('neu')}, pos={y_true.count('pos')})")

    print("\nLoading model …")
    tok  = AutoTokenizer.from_pretrained(tok_src)
    pipe = pipeline("text-classification", model=model_path, tokenizer=tok,
                    device=DEVICE, top_k=None)

    print("Collecting raw scores …")
    all_scores = collect_scores(pipe, texts)

    # Default macro-F1 (pure argmax, equivalent to T_neg=T_pos=0.5)
    default_preds = shifted_argmax(all_scores, 0.5, 0.5)
    default_f1 = f1_score(y_true, default_preds, labels=CLASSES, average="macro",
                          zero_division=0)
    print(f"\nDefault macro-F1 (T_neg=0.50, T_pos=0.50): {default_f1:.4f}")

    # ── 5-fold cross-validation ──────────────────────────────────────────────
    print(f"\n{N_FOLDS}-fold cross-validation …")
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=42)

    cv_default, cv_calib = [], []
    fold_thresholds = []

    for fold, (train_idx, test_idx) in enumerate(skf.split(texts, y_true)):
        train_s = [all_scores[i] for i in train_idx]
        test_s  = [all_scores[i] for i in test_idx]
        train_y = [y_true[i] for i in train_idx]
        test_y  = [y_true[i] for i in test_idx]

        t_neg, t_pos, _ = grid_search(train_s, train_y)

        calib_f1   = f1_score(test_y, shifted_argmax(test_s, t_neg, t_pos),
                              labels=CLASSES, average="macro", zero_division=0)
        default_f1_fold = f1_score(test_y, shifted_argmax(test_s, 0.5, 0.5),
                                   labels=CLASSES, average="macro", zero_division=0)

        cv_default.append(default_f1_fold)
        cv_calib.append(calib_f1)
        fold_thresholds.append((t_neg, t_pos))
        print(f"  Fold {fold+1}:  default={default_f1_fold:.4f}  "
              f"calibrated={calib_f1:.4f}  (T_neg={t_neg:.2f}, T_pos={t_pos:.2f})")

    print(f"\n  CV default  : {np.mean(cv_default):.4f} ± {np.std(cv_default):.4f}")
    print(f"  CV calibrated: {np.mean(cv_calib):.4f} ± {np.std(cv_calib):.4f}")
    print(f"  CV gain      : {np.mean(cv_calib) - np.mean(cv_default):+.4f}")

    # ── Global thresholds — fit on full dataset, for deployment ─────────────
    print("\nFitting thresholds on full dataset (for deployment) …")
    t_neg_g, t_pos_g, full_calib_f1 = grid_search(all_scores, y_true)

    print(f"  T_neg={t_neg_g:.2f}  T_pos={t_pos_g:.2f}  macro-F1={full_calib_f1:.4f}")
    print(f"  (vs default: {default_f1:.4f}, gain: {full_calib_f1 - default_f1:+.4f})")

    # Per-class breakdown with global thresholds
    calib_preds_global = shifted_argmax(all_scores, t_neg_g, t_pos_g)
    print("\n  Per-class with calibrated thresholds (full dataset):")
    report = classification_report(y_true, calib_preds_global, labels=CLASSES,
                                   zero_division=0, output_dict=True)
    print(f"  {'class':6s}  {'P':>6}  {'R':>6}  {'F1':>6}  {'n':>5}")
    print(f"  {'-'*36}")
    for c in CLASSES:
        r = report[c]
        print(f"  {c:6s}  {r['precision']:6.3f}  {r['recall']:6.3f}  "
              f"{r['f1-score']:6.3f}  {int(r['support']):5d}")

    print("\n  Per-class with DEFAULT thresholds (full dataset):")
    report_def = classification_report(y_true, default_preds, labels=CLASSES,
                                       zero_division=0, output_dict=True)
    print(f"  {'class':6s}  {'P':>6}  {'R':>6}  {'F1':>6}  {'n':>5}")
    print(f"  {'-'*36}")
    for c in CLASSES:
        r = report_def[c]
        print(f"  {c:6s}  {r['precision']:6.3f}  {r['recall']:6.3f}  "
              f"{r['f1-score']:6.3f}  {int(r['support']):5d}")

    # Save
    out = {
        "model": args.model,
        "dataset": "domaine_reel",
        "n": len(texts),
        "default_macro_f1": round(default_f1, 4),
        "calibrated_macro_f1_full": round(full_calib_f1, 4),
        "cv_default_mean": round(float(np.mean(cv_default)), 4),
        "cv_default_std":  round(float(np.std(cv_default)), 4),
        "cv_calib_mean":   round(float(np.mean(cv_calib)), 4),
        "cv_calib_std":    round(float(np.std(cv_calib)), 4),
        "cv_gain":         round(float(np.mean(cv_calib) - np.mean(cv_default)), 4),
        "thresholds": {
            "neg": t_neg_g,
            "neu": 0.50,
            "pos": t_pos_g,
        },
        "fold_thresholds": [{"neg": tn, "pos": tp} for tn, tp in fold_thresholds],
        "per_class_calibrated": {c: {k: round(report[c][k], 4)
                                     for k in ["precision", "recall", "f1-score", "support"]}
                                 for c in CLASSES},
        "per_class_default": {c: {k: round(report_def[c][k], 4)
                                  for k in ["precision", "recall", "f1-score", "support"]}
                              for c in CLASSES},
    }
    out_path = f"results/thresholds_{args.model}.json"
    os.makedirs("results", exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=2)
    print(f"\n→ {out_path}")


if __name__ == "__main__":
    main()
