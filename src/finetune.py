"""
Step 4 — Fine-tune encoder models with a 3-class sentiment head.

Config (per plan):
  fp16, batch=16, max_len=128, lr=2e-5, 3 epochs,
  eval per epoch, best model selected by macro-F1.

Usage:
    python src/finetune.py --model darijabert
    python src/finetune.py --model darijabert-arabizi
    python src/finetune.py --model marbertv2
    python src/finetune.py --model qarib
"""

import argparse
import os
import time

import numpy as np
import torch
from datasets import Dataset
from sklearn.model_selection import train_test_split
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
)

import pandas as pd
from sklearn.metrics import f1_score
from utils import set_seeds, SEED
from harness import evaluate_model, free_gpu

set_seeds()

DATA_DIR   = os.path.join(os.path.dirname(__file__), "..", "data", "test_sets")
RAW_DIR    = os.path.join(os.path.dirname(__file__), "..", "data", "raw")
CKPT_DIR   = os.path.join(os.path.dirname(__file__), "..", "checkpoints")
os.makedirs(CKPT_DIR, exist_ok=True)

LABEL2ID = {"neg": 0, "neu": 1, "pos": 2}
ID2LABEL  = {v: k for k, v in LABEL2ID.items()}

# ── Model registry ────────────────────────────────────────────────────────
# qarib: verify current canonical ID at https://huggingface.co/qarib
MODELS = {
    "darijabert":         {"hub_id": "SI2M-Lab/DarijaBERT",         "train_lang": "darija_ar", "eval_langs": ["darija_ar"]},
    "darijabert-arabizi": {"hub_id": "SI2M-Lab/DarijaBERT-arabizi", "train_lang": "arabizi",  "eval_langs": ["arabizi"]},
    "marbertv2":          {"hub_id": "UBC-NLP/MARBERTv2",           "train_lang": "darija_ar", "eval_langs": ["darija_ar", "msa"],
                           "license_note": "RESEARCH-ONLY — not for commercial use."},
    "qarib":              {"hub_id": "qarib/bert-base-qarib",        "train_lang": "darija_ar", "eval_langs": ["darija_ar", "msa"]},
}


def load_train_split(lang: str):
    """Load test set CSV and carve out a train split (80/20 of the non-test rows).

    Note: the test set is the 1000-row frozen file.  We load the raw data for
    training; the frozen test set is ONLY used for evaluation.
    """
    # For training data, load the full raw dataset (not the frozen test set)
    raw_map = {
        "darija_ar": os.path.join(RAW_DIR, "MAC"),
        "arabizi":   os.path.join(RAW_DIR, "MYC"),
        "msa":       os.path.join(RAW_DIR, "ASTD"),
    }
    raw_path = raw_map.get(lang)
    if raw_path is None or not os.path.isdir(raw_path):
        raise FileNotFoundError(
            f"Raw training data not found at {raw_path}. "
            "Run build_test_sets.py first and ensure raw data is present."
        )

    # Re-use the same loaders from build_test_sets
    import importlib, sys
    sys.path.insert(0, os.path.dirname(__file__))
    bts = importlib.import_module("build_test_sets")
    importlib.reload(bts)

    if lang == "darija_ar":
        df = bts.load_mac()
    elif lang == "arabizi":
        df = bts.load_myc()
    elif lang == "msa":
        df = bts.load_astd()
    else:
        raise ValueError(f"Unknown lang: {lang}")

    # Exclude the frozen test rows to avoid leakage
    test_df = pd.read_csv(os.path.join(DATA_DIR, f"{lang}.csv"))
    test_texts = set(test_df["text"].tolist())
    df = df[~df["text"].isin(test_texts)]

    train_df, val_df = train_test_split(
        df, test_size=0.1, stratify=df["label"], random_state=SEED
    )
    return train_df.reset_index(drop=True), val_df.reset_index(drop=True)


def encode(df: pd.DataFrame, tokenizer, max_len=128) -> Dataset:
    df = df.copy()
    df["text"] = df["text"].fillna("").astype(str)   # guard against NaN rows
    df = df[df["label"].isin(LABEL2ID)]               # drop any unmapped labels
    ds = Dataset.from_pandas(df, preserve_index=False)

    def tokenize(batch):
        out = tokenizer(
            batch["text"], truncation=True, padding="max_length", max_length=max_len
        )
        out["labels"] = [LABEL2ID[l] for l in batch["label"]]
        return out

    return ds.map(tokenize, batched=True, remove_columns=ds.column_names)


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    classes = sorted(set(labels.tolist()))
    f1 = f1_score(labels, preds, labels=classes, average="macro", zero_division=0)
    return {"macro_f1": f1}


def finetune(model_key: str) -> None:
    cfg = MODELS[model_key]
    hub_id    = cfg["hub_id"]
    train_lang = cfg["train_lang"]
    eval_langs = cfg["eval_langs"]

    if "license_note" in cfg:
        print(f"  [LICENSE] {cfg['license_note']}")

    print(f"\n=== Fine-tuning {model_key} ({hub_id}) on {train_lang} ===")
    t_start = time.time()

    tokenizer = AutoTokenizer.from_pretrained(hub_id)
    train_df, val_df = load_train_split(train_lang)
    print(f"  train={len(train_df)}  val={len(val_df)}")

    train_ds = encode(train_df, tokenizer)
    val_ds   = encode(val_df,   tokenizer)

    model = AutoModelForSequenceClassification.from_pretrained(
        hub_id,
        num_labels=3,
        id2label=ID2LABEL,
        label2id=LABEL2ID,
        ignore_mismatched_sizes=True,
    )

    ckpt_path = os.path.join(CKPT_DIR, model_key)
    training_args = TrainingArguments(
        output_dir=ckpt_path,
        num_train_epochs=3,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=32,
        learning_rate=2e-5,
        fp16=torch.cuda.is_available(),
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="macro_f1",
        greater_is_better=True,
        logging_steps=50,
        seed=SEED,
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        compute_metrics=compute_metrics,
    )

    trainer.train()
    trainer.save_model(ckpt_path)
    tokenizer.save_pretrained(ckpt_path)

    gpu_minutes = (time.time() - t_start) / 60
    print(f"  Training done in {gpu_minutes:.1f} min (GPU time cost).")

    # Evaluate through the shared harness
    from transformers import pipeline as hf_pipeline
    from label_maps import FINETUNED_MAP, apply_map

    pipe = hf_pipeline(
        "text-classification",
        model=ckpt_path,
        tokenizer=ckpt_path,
        device=0 if torch.cuda.is_available() else -1,
        top_k=None,
    )

    def predict_fn(texts):
        results = pipe(texts, batch_size=32, truncation=True)
        preds = []
        for item in results:
            scores = item if isinstance(item, list) else [item]
            best = max(scores, key=lambda x: x["score"])
            preds.append(apply_map(best["label"], FINETUNED_MAP))
        return preds

    for lang in eval_langs:
        evaluate_model(
            model_key, lang, predict_fn, model_obj=model,
            extra_meta={"gpu_train_minutes": round(gpu_minutes, 1),
                        "checkpoint": ckpt_path},
        )

    free_gpu(model)
    del pipe, model


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model", required=True, choices=list(MODELS),
        help="Which model to fine-tune",
    )
    args = parser.parse_args()
    finetune(args.model)
