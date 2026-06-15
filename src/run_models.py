"""
Step 3 — Run the three ready-made models through the benchmark harness.

Models:
  xlm-t           cardiffnlp/twitter-xlm-roberta-base-sentiment  (ALL langs)
  camelbert-da    CAMeL-Lab/bert-base-arabic-camelbert-da-sentiment (darija_ar + msa)
  distilcamembert cmarkea/distilcamembert-base-sentiment          (francais)

Usage:
    python src/run_models.py [--models xlm-t camelbert-da distilcamembert]
"""

import argparse
import sys
import os

import torch
from transformers import pipeline, AutoConfig

from utils import set_seeds, SEED
from label_maps import (
    XLM_T_MAP, CAMELBERT_DA_MAP, CAMELBERT_NEU_THRESHOLD,
    DISTILCAMEMBERT_MAP, FINETUNED_MAP, apply_map,
)
from harness import evaluate_model, free_gpu

set_seeds()

DEVICE = 0 if torch.cuda.is_available() else -1
BATCH_SIZE = 32


# ── helpers ───────────────────────────────────────────────────────────────

def make_pipeline_predict(pipe, label_map, neu_threshold=None):
    """Return a predict_fn(texts) -> List[canonical_label]."""
    def predict(texts):
        results = pipe(texts, batch_size=BATCH_SIZE, truncation=True, max_length=512)
        preds = []
        for item in results:
            # top_k=None returns list[dict] per input in a batch
            scores = item if isinstance(item, list) else [item]
            best = max(scores, key=lambda x: x["score"])
            if neu_threshold is not None and best["score"] < neu_threshold:
                preds.append("neu")
            else:
                preds.append(apply_map(best["label"], label_map))
        return preds
    return predict


# ── xlm-t ─────────────────────────────────────────────────────────────────

def run_xlm_t(langs):
    model_id = "cardiffnlp/twitter-xlm-roberta-base-sentiment"
    print(f"\n=== xlm-t ({model_id}) ===")
    cfg = AutoConfig.from_pretrained(model_id)
    print(f"  id2label: {cfg.id2label}")

    pipe = pipeline(
        "text-classification", model=model_id, device=DEVICE,
        top_k=None,
    )
    predict_fn = make_pipeline_predict(pipe, XLM_T_MAP)

    for lang in langs:
        evaluate_model("xlm-t", lang, predict_fn, model_obj=pipe.model)

    free_gpu(pipe.model)
    del pipe


# ── camelbert-da ──────────────────────────────────────────────────────────

def run_camelbert_da(langs):
    model_id = "CAMeL-Lab/bert-base-arabic-camelbert-da-sentiment"
    print(f"\n=== camelbert-da ({model_id}) ===")
    cfg = AutoConfig.from_pretrained(model_id)
    print(f"  id2label: {cfg.id2label}")
    print(f"  Binary model — threshold τ={CAMELBERT_NEU_THRESHOLD} for neutral assignment")

    pipe = pipeline(
        "text-classification", model=model_id, device=DEVICE,
        top_k=None,
    )
    predict_fn = make_pipeline_predict(pipe, CAMELBERT_DA_MAP, CAMELBERT_NEU_THRESHOLD)

    for lang in langs:
        evaluate_model(
            "camelbert-da", lang, predict_fn, model_obj=pipe.model,
            extra_meta={"neu_threshold": CAMELBERT_NEU_THRESHOLD, "binary_original": True},
        )

    free_gpu(pipe.model)
    del pipe


# ── distilcamembert ───────────────────────────────────────────────────────

def run_distilcamembert(langs):
    model_id = "cmarkea/distilcamembert-base-sentiment"
    print(f"\n=== distilcamembert ({model_id}) ===")
    cfg = AutoConfig.from_pretrained(model_id)
    print(f"  id2label: {cfg.id2label}")
    print("  5-star model — mapping: 1-2*->neg, 3*->neu, 4-5*->pos")

    pipe = pipeline(
        "text-classification", model=model_id, device=DEVICE,
        top_k=None,
    )
    predict_fn = make_pipeline_predict(pipe, DISTILCAMEMBERT_MAP)

    for lang in langs:
        evaluate_model("distilcamembert", lang, predict_fn, model_obj=pipe.model)

    free_gpu(pipe.model)
    del pipe


# ── main ──────────────────────────────────────────────────────────────────

ALL_LANGS = ["darija_ar", "francais", "msa", "arabizi"]

MODEL_LANG_MAP = {
    "xlm-t":           ALL_LANGS,
    "camelbert-da":    ["darija_ar", "msa"],
    "distilcamembert": ["francais"],
}

RUNNERS = {
    "xlm-t":           run_xlm_t,
    "camelbert-da":    run_camelbert_da,
    "distilcamembert": run_distilcamembert,
}

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--models", nargs="+", default=list(RUNNERS),
        choices=list(RUNNERS),
        help="Which models to run (default: all)",
    )
    args = parser.parse_args()

    for model_key in args.models:
        langs = MODEL_LANG_MAP[model_key]
        RUNNERS[model_key](langs)

    print("\nDone. Results written to results/.")
    print("Run src/aggregate.py to produce plots and the weighted grid.")
