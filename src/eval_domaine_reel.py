"""
Evaluate all locally-available Arabic models on the domaine-réel test set.

Usage:
    python src/eval_domaine_reel.py                           # all models, default CSV
    python src/eval_domaine_reel.py --models marbertv2 qarib
    python src/eval_domaine_reel.py --test-csv data/test_sets/gemini.csv
"""

import argparse
import gc
import json
import os
import sys
import time

import torch
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from transformers import pipeline, AutoTokenizer

sys.path.insert(0, os.path.dirname(__file__))
from label_maps import (
    XLM_T_MAP, CAMELBERT_DA_MAP, FINETUNED_MAP, apply_map,
)

TEST_CSV = "data/test_sets/domaine_reel.csv"
DEVICE   = 0 if torch.cuda.is_available() else -1
BATCH    = 16

# ── model registry ─────────────────────────────────────────────────────────
# Each entry: (model_path_or_hub_id, tokenizer_source, label_map)
# tokenizer_source = None means use the model path directly
MODELS = {
    "marbertv2":   ("checkpoints/marbertv2/checkpoint-1311",   "UBC-NLP/MARBERTv2",              FINETUNED_MAP),
    "qarib":       ("checkpoints/qarib/checkpoint-1311",        "qarib/bert-base-qarib",          FINETUNED_MAP),
    "darijabert":  ("checkpoints/darijabert/checkpoint-1311",   "SI2M-Lab/DarijaBERT",            FINETUNED_MAP),
    "camelbert-da":("CAMeL-Lab/bert-base-arabic-camelbert-da-sentiment", None,                    CAMELBERT_DA_MAP),
    "xlm-t":       ("cardiffnlp/twitter-xlm-roberta-base-sentiment",     None,                    XLM_T_MAP),
}

# Public darija_ar results for gap comparison
PUBLIC_F1 = {
    "marbertv2":    0.8441,
    "qarib":        0.8265,
    "darijabert":   0.7745,
    "camelbert-da": 0.7008,
    "xlm-t":        0.7230,
}


def run_model(name: str, test_csv: str = TEST_CSV) -> dict:
    model_path, tok_src, label_map = MODELS[name]
    print(f"\n{'='*60}")
    print(f"  {name}  ({'local ckpt' if os.path.exists(model_path) else 'HF hub'})")
    print(f"{'='*60}")

    tok_source = tok_src if tok_src else model_path
    tok = AutoTokenizer.from_pretrained(tok_source)
    pipe = pipeline("text-classification", model=model_path, tokenizer=tok,
                    device=DEVICE, top_k=None)

    df     = pd.read_csv(test_csv)
    texts  = df["text"].tolist()
    y_true = df["label"].tolist()

    # warm-up
    _ = pipe(texts[:3], batch_size=BATCH, truncation=True, max_length=512)

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    t0 = time.perf_counter()
    results = pipe(texts, batch_size=BATCH, truncation=True, max_length=512)
    elapsed = time.perf_counter() - t0

    y_pred = []
    for item in results:
        scores = item if isinstance(item, list) else [item]
        best   = max(scores, key=lambda x: x["score"])
        y_pred.append(apply_map(best["label"], label_map))

    latency_ms = (elapsed / len(texts)) * 1000
    peak_vram  = torch.cuda.max_memory_allocated() / 1024**2 if torch.cuda.is_available() else 0.0

    classes  = sorted(set(y_true))
    macro_f1 = f1_score(y_true, y_pred, labels=classes, average="macro", zero_division=0)
    report   = classification_report(y_true, y_pred, labels=classes,
                                     zero_division=0, output_dict=True)
    cm       = confusion_matrix(y_true, y_pred, labels=classes).tolist()

    # Print results
    print(f"  macro-F1 : {macro_f1:.4f}  "
          f"(public darija: {PUBLIC_F1.get(name,'?'):.4f}  "
          f"gap: {macro_f1 - PUBLIC_F1.get(name, macro_f1):+.4f})")
    print(f"  latency  : {latency_ms:.1f} ms/utt")
    print(f"\n  {'class':6s}  {'P':>6}  {'R':>6}  {'F1':>6}  {'n':>5}")
    print(f"  {'-'*36}")
    for c in classes:
        r = report[c]
        print(f"  {c:6s}  {r['precision']:6.3f}  {r['recall']:6.3f}  "
              f"{r['f1-score']:6.3f}  {int(r['support']):5d}")
    print(f"\n  Confusion matrix (rows=true, cols=pred, labels={classes}):")
    for row in cm:
        print(f"    {row}")

    metrics = {
        "model": name, "lang": "domaine_reel",
        "n": len(texts), "macro_f1": round(macro_f1, 4),
        "latency_ms_per_utt": round(latency_ms, 3),
        "peak_vram_mb": round(peak_vram, 1),
        "classes_evaluated": classes,
        "classification_report": report,
        "confusion_matrix": cm,
        "confusion_matrix_labels": classes,
    }
    tag = os.path.splitext(os.path.basename(test_csv))[0]
    out = f"results/{name}_{tag}.json"
    os.makedirs("results", exist_ok=True)
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(metrics, fh, ensure_ascii=False, indent=2)
    print(f"\n  → {out}")

    del pipe
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", default=list(MODELS),
                        choices=list(MODELS))
    parser.add_argument("--test-csv", default=TEST_CSV,
                        help="Path to labelled test CSV (must have text + label columns)")
    args = parser.parse_args()

    all_results = []
    for name in args.models:
        m = run_model(name, test_csv=args.test_csv)
        all_results.append(m)

    # Summary table
    print(f"\n{'='*60}")
    print("SUMMARY — Domaine réel vs public darija_ar")
    print(f"{'='*60}")
    print(f"  {'Model':15s}  {'Public':>8}  {'Domaine':>8}  {'Gap':>7}")
    print(f"  {'-'*46}")
    for m in sorted(all_results, key=lambda x: -x["macro_f1"]):
        pub = PUBLIC_F1.get(m["model"], float("nan"))
        gap = m["macro_f1"] - pub
        print(f"  {m['model']:15s}  {pub:8.4f}  {m['macro_f1']:8.4f}  {gap:+7.4f}")


if __name__ == "__main__":
    main()
