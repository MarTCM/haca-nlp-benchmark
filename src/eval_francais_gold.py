"""
Evaluate a French tonality model on the frozen French gold (francais_haca_gold.csv).

Works for any model key registered in haca_pipeline.MODEL_REGISTRY (or any checkpoints/<key>):
the off-the-shelf baselines (`xlm-sentiment`, `distilcamembert`) and the HACA fine-tunes
(`camembert-haca`, `xlm-r-haca`) — so you can measure exactly what the fine-tune buys.

Usage:
    python src/eval_francais_gold.py --models xlm-sentiment distilcamembert
    python src/eval_francais_gold.py --models xlm-sentiment camembert-haca xlm-r-haca
"""

import argparse
import os
import sys

import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, f1_score

sys.path.insert(0, os.path.dirname(__file__))
import haca_pipeline as hp  # noqa: E402

GOLD = "data/test_sets/francais_haca_gold.csv"
CLASSES = ["neg", "neu", "pos"]


def eval_model(model_key: str, texts, y_true):
    predict_proba, thr = hp.load_encoder(model_key)
    probas = predict_proba(list(texts))
    y_pred = [hp.shifted_argmax(p, thr) for p in probas]
    macro = f1_score(y_true, y_pred, labels=CLASSES, average="macro", zero_division=0)
    print(f"\n=== {model_key} — macro-F1 = {macro:.3f} ===")
    print(classification_report(y_true, y_pred, labels=CLASSES, zero_division=0, digits=3))
    print("confusion (rows=true neg/neu/pos, cols=pred):")
    print(confusion_matrix(y_true, y_pred, labels=CLASSES))
    return macro


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=["xlm-sentiment"],
                    help="model keys to evaluate (registry keys or checkpoints/<key>)")
    ap.add_argument("--gold", default=GOLD)
    args = ap.parse_args()

    df = pd.read_csv(args.gold)
    texts, y_true = df["text"].astype(str).tolist(), df["label"].tolist()
    print(f"gold: {len(df)} utterances  dist={dict(df['label'].value_counts())}")

    scores = {}
    for m in args.models:
        try:
            scores[m] = eval_model(m, texts, y_true)
        except Exception as e:
            print(f"\n=== {m} — FAILED: {e} ===")
    if scores:
        print("\nsummary (macro-F1):")
        for m, s in sorted(scores.items(), key=lambda kv: -kv[1]):
            print(f"  {s:.3f}  {m}")


if __name__ == "__main__":
    main()
