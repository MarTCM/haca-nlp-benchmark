"""
Benchmark harness — run once per (model × language).

Produces:
  results/<model>_<lang>.json    — per-run metrics
  results/summary.csv            — aggregated table (appended each run)

Usage:
    from harness import evaluate_model
"""

import gc
import json
import os
import time
from typing import Callable, List

import numpy as np
import pandas as pd
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
)

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

SUMMARY_CSV = os.path.join(RESULTS_DIR, "summary.csv")
WARMUP = 5


def _param_count(model) -> int:
    try:
        return sum(p.numel() for p in model.parameters())
    except Exception:
        return -1


def _peak_vram_mb() -> float:
    try:
        import torch
        if torch.cuda.is_available():
            return torch.cuda.max_memory_allocated() / 1024 ** 2
    except ImportError:
        pass
    return 0.0


def _reset_vram_counter() -> None:
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
    except ImportError:
        pass


def free_gpu(model=None) -> None:
    """Release GPU memory between model runs."""
    if model is not None:
        try:
            del model
        except Exception:
            pass
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass


def evaluate_model(
    model_name: str,
    lang: str,
    predict_fn: Callable[[List[str]], List[str]],
    model_obj=None,
    extra_meta: dict | None = None,
) -> dict:
    """
    Parameters
    ----------
    model_name  : short identifier, e.g. "xlm-t"
    lang        : one of darija_ar / francais / msa / arabizi
    predict_fn  : callable(texts: List[str]) -> List[str canonical label]
    model_obj   : optional reference used only for param counting / VRAM
    extra_meta  : dict of additional fields added to the JSON (e.g. non_response_rate)

    Returns the metrics dict and writes results/<model>_<lang>.json.
    """
    test_path = os.path.join(
        os.path.dirname(__file__), "..", "data", "test_sets", f"{lang}.csv"
    )
    if not os.path.exists(test_path):
        raise FileNotFoundError(
            f"Test set not found: {test_path}. Run build_test_sets.py first."
        )

    df = pd.read_csv(test_path)
    texts = df["text"].tolist()
    y_true = df["label"].tolist()

    _reset_vram_counter()

    # Warm-up
    _ = predict_fn(texts[:WARMUP])

    # Timed inference
    t0 = time.perf_counter()
    y_pred = predict_fn(texts)
    elapsed = time.perf_counter() - t0

    latency_ms = (elapsed / len(texts)) * 1000  # ms per utterance

    classes = sorted(set(y_true))  # only classes present in ground truth
    macro_f1 = f1_score(y_true, y_pred, labels=classes, average="macro", zero_division=0)
    report = classification_report(
        y_true, y_pred, labels=classes, zero_division=0, output_dict=True
    )
    cm = confusion_matrix(y_true, y_pred, labels=classes).tolist()
    peak_vram = _peak_vram_mb()
    n_params = _param_count(model_obj) if model_obj is not None else -1

    metrics = {
        "model": model_name,
        "lang": lang,
        "n": len(texts),
        "macro_f1": round(macro_f1, 4),
        "latency_ms_per_utt": round(latency_ms, 3),
        "peak_vram_mb": round(peak_vram, 1),
        "n_params": n_params,
        "classes_evaluated": classes,
        "classification_report": report,
        "confusion_matrix": cm,
        "confusion_matrix_labels": classes,
    }
    if extra_meta:
        metrics.update(extra_meta)

    out_path = os.path.join(RESULTS_DIR, f"{model_name}_{lang}.json")
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(metrics, fh, ensure_ascii=False, indent=2)

    # Append to summary CSV
    row = {
        "model": model_name,
        "lang": lang,
        "macro_f1": macro_f1,
        "latency_ms": latency_ms,
        "peak_vram_mb": peak_vram,
        "n_params": n_params,
        "n": len(texts),
    }
    if extra_meta:
        row.update(extra_meta)
    summary_row = pd.DataFrame([row])
    if os.path.exists(SUMMARY_CSV):
        existing = pd.read_csv(SUMMARY_CSV)
        # Replace row if (model, lang) already present
        existing = existing[~((existing["model"] == model_name) & (existing["lang"] == lang))]
        summary_row = pd.concat([existing, summary_row], ignore_index=True)
    summary_row.to_csv(SUMMARY_CSV, index=False)

    print(
        f"  [{model_name} × {lang}]  macro-F1={macro_f1:.4f}  "
        f"latency={latency_ms:.1f}ms  VRAM={peak_vram:.0f}MB"
    )
    return metrics
