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
from sklearn.utils.class_weight import compute_class_weight
from utils import set_seeds, SEED
from harness import evaluate_model, free_gpu

set_seeds()

DATA_DIR        = os.path.join(os.path.dirname(__file__), "..", "data", "test_sets")
RAW_DIR         = os.path.join(os.path.dirname(__file__), "..", "data", "raw")
CKPT_DIR        = os.path.join(os.path.dirname(__file__), "..", "checkpoints")
BROADCAST_TRAIN = os.path.join(DATA_DIR, "broadcast_train_raw.csv")
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
    # ── Option 2: broadcast-aware models ──────────────────────────────────
    # marbertv2-mixed: fresh from Hub, trained on MAC + broadcast (content-valence labels).
    # marbertv2-broadcast: starts from the existing MAC checkpoint, adapts with broadcast only.
    "marbertv2-mixed": {
        "hub_id":            "UBC-NLP/MARBERTv2",
        "train_lang":        "mixed",
        "eval_langs":        ["darija_ar", "domaine_reel_v2"],
        "license_note":      "RESEARCH-ONLY — not for commercial use.",
        "lr":                2e-5,
        "epochs":            3,
        "batch_size":        16,
        "use_class_weights": True,
    },
    "marbertv2-broadcast": {
        "hub_id":            None,                   # load from local checkpoint
        "local_ckpt":        "checkpoints/marbertv2",
        "train_lang":        "broadcast",
        "eval_langs":        ["domaine_reel_v2", "darija_ar"],
        "license_note":      "RESEARCH-ONLY — not for commercial use.",
        "lr":                5e-6,                   # low LR to avoid forgetting MAC
        "epochs":            5,
        "batch_size":        8,
        "use_class_weights": True,
    },
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


def _load_broadcast_df() -> pd.DataFrame:
    """Load broadcast_train_raw.csv and return a [text, label] DataFrame."""
    if not os.path.exists(BROADCAST_TRAIN):
        raise FileNotFoundError(
            f"Broadcast training data not found: {BROADCAST_TRAIN}\n"
            "Run:  python src/annotate_gemini.py  (requires GEMINI_API_KEY)"
        )
    df = pd.read_csv(BROADCAST_TRAIN)
    df = df[df["label"].isin(LABEL2ID)].reset_index(drop=True)
    return df[["text", "label"]]


def load_train_split_mixed():
    """MAC training data + broadcast training data, combined with a joint 80/20 split."""
    import importlib, sys as _sys
    _sys.path.insert(0, os.path.dirname(__file__))
    bts = importlib.import_module("build_test_sets")
    importlib.reload(bts)

    mac_df = bts.load_mac()
    test_df = pd.read_csv(os.path.join(DATA_DIR, "darija_ar.csv"))
    mac_df  = mac_df[~mac_df["text"].isin(set(test_df["text"]))].reset_index(drop=True)

    broadcast_df = _load_broadcast_df()
    # Exclude the frozen domaine_reel_v2 test rows from training
    frozen_v2 = pd.read_csv(os.path.join(DATA_DIR, "domaine_reel_v2.csv"))
    broadcast_df = broadcast_df[~broadcast_df["text"].isin(set(frozen_v2["text"]))].reset_index(drop=True)

    # Combine and split each source separately to preserve stratification, then merge
    mac_tr, mac_va = train_test_split(mac_df, test_size=0.1, stratify=mac_df["label"], random_state=SEED)
    bc_tr,  bc_va  = train_test_split(broadcast_df, test_size=0.2, stratify=broadcast_df["label"], random_state=SEED)

    train_df = pd.concat([mac_tr, bc_tr], ignore_index=True).sample(frac=1, random_state=SEED)
    val_df   = pd.concat([mac_va, bc_va], ignore_index=True).sample(frac=1, random_state=SEED)

    print(f"  mixed train: {len(train_df)}  (MAC={len(mac_tr)}, broadcast={len(bc_tr)})")
    print(f"  mixed val  : {len(val_df)}   (MAC={len(mac_va)}, broadcast={len(bc_va)})")
    print(f"  train dist : {dict(train_df['label'].value_counts())}")
    return train_df.reset_index(drop=True), val_df.reset_index(drop=True)


def load_train_split_broadcast():
    """Broadcast data only: 80/20 split.  Validation uses MAC val for stable early stopping."""
    import importlib, sys as _sys
    _sys.path.insert(0, os.path.dirname(__file__))
    bts = importlib.import_module("build_test_sets")
    importlib.reload(bts)

    broadcast_df = _load_broadcast_df()
    frozen_v2    = pd.read_csv(os.path.join(DATA_DIR, "domaine_reel_v2.csv"))
    broadcast_df = broadcast_df[~broadcast_df["text"].isin(set(frozen_v2["text"]))].reset_index(drop=True)

    bc_tr, bc_va = train_test_split(broadcast_df, test_size=0.2, stratify=broadcast_df["label"], random_state=SEED)

    # Augment validation with MAC val for a more stable macro-F1 signal during training
    mac_df  = bts.load_mac()
    test_df = pd.read_csv(os.path.join(DATA_DIR, "darija_ar.csv"))
    mac_df  = mac_df[~mac_df["text"].isin(set(test_df["text"]))].reset_index(drop=True)
    _, mac_va = train_test_split(mac_df, test_size=0.1, stratify=mac_df["label"], random_state=SEED)

    val_df = pd.concat([bc_va, mac_va], ignore_index=True).sample(frac=1, random_state=SEED)

    print(f"  broadcast train: {len(bc_tr)}  val: {len(val_df)} (bc={len(bc_va)}, mac={len(mac_va)})")
    print(f"  train dist: {dict(bc_tr['label'].value_counts())}")
    return bc_tr.reset_index(drop=True), val_df.reset_index(drop=True)


def compute_class_weights_tensor(train_df: pd.DataFrame) -> torch.Tensor:
    """Return a FloatTensor of class weights (inverse frequency, sklearn-style)."""
    labels = train_df["label"].map(LABEL2ID).values
    classes = sorted(LABEL2ID.values())
    weights = compute_class_weight("balanced", classes=np.array(classes), y=labels)
    return torch.tensor(weights, dtype=torch.float32)


class WeightedTrainer(Trainer):
    """Trainer subclass that applies per-class weights to the cross-entropy loss."""

    def __init__(self, *args, class_weights: torch.Tensor | None = None, **kwargs):
        super().__init__(*args, **kwargs)
        self._cw = class_weights

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels  = inputs.get("labels")
        outputs = model(**inputs)
        if self._cw is not None and labels is not None:
            logits = outputs.logits
            weight = self._cw.to(logits.device, dtype=logits.dtype)
            loss   = torch.nn.functional.cross_entropy(logits, labels, weight=weight)
        else:
            loss = outputs.loss
        return (loss, outputs) if return_outputs else loss


def encode(df: pd.DataFrame, tokenizer, max_len=128) -> Dataset:
    df = df.copy()
    df["text"] = df["text"].fillna("").astype(str)   # guard against NaN rows
    df = df[df["label"].isin(LABEL2ID)]               # drop any unmapped labels
    ds = Dataset.from_pandas(df, preserve_index=False)

    def tokenize(batch):
        texts = [str(t) for t in batch["text"]]   # force plain Python str (not Arrow array)
        out = tokenizer(
            texts, truncation=True, padding="max_length", max_length=max_len
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
    cfg        = MODELS[model_key]
    hub_id     = cfg["hub_id"]
    local_ckpt = cfg.get("local_ckpt")
    train_lang = cfg["train_lang"]
    eval_langs = cfg["eval_langs"]

    # Per-model hyper-parameter overrides (fall back to plan defaults)
    lr         = cfg.get("lr",         2e-5)
    epochs     = cfg.get("epochs",     3)
    batch_size = cfg.get("batch_size", 16)
    use_cw     = cfg.get("use_class_weights", False)

    if "license_note" in cfg:
        print(f"  [LICENSE] {cfg['license_note']}")

    model_source = local_ckpt if hub_id is None else hub_id
    print(f"\n=== Fine-tuning {model_key}  source={model_source}  "
          f"train_lang={train_lang}  lr={lr}  epochs={epochs}  batch={batch_size} ===")
    t_start = time.time()

    # ── Load tokenizer ────────────────────────────────────────────────────────
    tok_source = local_ckpt if hub_id is None else hub_id
    tokenizer  = AutoTokenizer.from_pretrained(tok_source)

    # ── Load training data ────────────────────────────────────────────────────
    if train_lang == "mixed":
        train_df, val_df = load_train_split_mixed()
    elif train_lang == "broadcast":
        train_df, val_df = load_train_split_broadcast()
    else:
        train_df, val_df = load_train_split(train_lang)
    print(f"  train={len(train_df)}  val={len(val_df)}")

    train_ds = encode(train_df, tokenizer)
    val_ds   = encode(val_df,   tokenizer)

    # ── Load model ────────────────────────────────────────────────────────────
    model_path = local_ckpt if hub_id is None else hub_id
    model = AutoModelForSequenceClassification.from_pretrained(
        model_path,
        num_labels=3,
        id2label=ID2LABEL,
        label2id=LABEL2ID,
        ignore_mismatched_sizes=True,
    )

    # ── Class weights ─────────────────────────────────────────────────────────
    cw_tensor = compute_class_weights_tensor(train_df) if use_cw else None
    if cw_tensor is not None:
        print(f"  class weights: neg={cw_tensor[0]:.3f}  "
              f"neu={cw_tensor[1]:.3f}  pos={cw_tensor[2]:.3f}")

    ckpt_path = os.path.join(CKPT_DIR, model_key)
    training_args = TrainingArguments(
        output_dir=ckpt_path,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=32,
        learning_rate=lr,
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

    TrainerClass = WeightedTrainer if use_cw else Trainer
    trainer_kwargs: dict = dict(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        compute_metrics=compute_metrics,
    )
    if use_cw:
        trainer_kwargs["class_weights"] = cw_tensor

    trainer = TrainerClass(**trainer_kwargs)

    trainer.train()
    trainer.save_model(ckpt_path)
    tokenizer.save_pretrained(ckpt_path)

    gpu_minutes = (time.time() - t_start) / 60
    print(f"  Training done in {gpu_minutes:.1f} min (GPU time cost).")

    # Evaluate through the shared harness
    from transformers import pipeline as hf_pipeline
    from transformers import AutoTokenizer as _AutoTokenizer
    from label_maps import FINETUNED_MAP, apply_map

    _tokenizer = _AutoTokenizer.from_pretrained(ckpt_path)
    _tokenizer.model_max_length = 512  # pipeline ignores max_length kwarg; patch tokenizer directly

    pipe = hf_pipeline(
        "text-classification",
        model=ckpt_path,
        tokenizer=_tokenizer,
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
        help="Which model to fine-tune  "
             "(marbertv2-mixed / marbertv2-broadcast for Option 2)",
    )
    args = parser.parse_args()
    finetune(args.model)
