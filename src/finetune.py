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
HACA_TRAIN      = os.path.join(DATA_DIR, "haca_train_v3.csv")   # Stage 4 output (v3)
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
    # ── Stage 5 (v3): HACA-Sent — MAC + cleaned/labelled SRT pool + synthetic ──
    # Consumes data/test_sets/haca_train_v3.csv (build_haca_train.py).
    # Class-weighted focal loss to fight the neu-dominant / pos-scarce imbalance.
    "marbertv2-haca": {
        "hub_id":            "UBC-NLP/MARBERTv2",
        "train_lang":        "haca",
        "eval_langs":        ["domaine_reel_v2", "darija_ar"],
        "license_note":      "RESEARCH-ONLY — not for commercial use.",
        "lr":                2e-5,
        "epochs":            4,
        "batch_size":        16,
        "use_class_weights": True,
        "focal_gamma":       2.0,
        "pos_oversample":    3,      # repeat in-domain pos rows to amplify the rare class
    },
    # In-domain ONLY (no MAC) — the fix for the pos-collapse: class weights and pos
    # examples both reflect broadcast content-valence. Continues from the MAC-trained
    # checkpoint to keep general Darija, with a low LR.
    "marbertv2-haca-only": {
        "hub_id":            None,
        "local_ckpt":        "checkpoints/marbertv2-haca",  # sharpen the model you already trained
        "train_lang":        "haca-only",
        "eval_langs":        ["domaine_reel_v2", "darija_ar"],
        "license_note":      "RESEARCH-ONLY — not for commercial use.",
        "lr":                1e-5,
        "epochs":            6,
        "batch_size":        16,
        "use_class_weights": True,
        "focal_gamma":       2.0,
        "pos_oversample":    4,
    },
    # Same, but fresh from Hub (use if checkpoints/marbertv2 isn't available this session).
    "marbertv2-haca-only-hub": {
        "hub_id":            "UBC-NLP/MARBERTv2",
        "train_lang":        "haca-only",
        "eval_langs":        ["domaine_reel_v2", "darija_ar"],
        "license_note":      "RESEARCH-ONLY — not for commercial use.",
        "lr":                2e-5,
        "epochs":            6,
        "batch_size":        16,
        "use_class_weights": True,
        "focal_gamma":       2.0,
        "pos_oversample":    4,
    },
    # ── French HACA-Sent — fine-tune a French encoder on hand-authored broadcast French ──
    # Training pool: src/synthetic_haca_fr.py (no real French pool exists). Eval: the frozen
    # real gold data/test_sets/francais_haca_gold.csv (run src/eval_francais_gold.py after).
    # Two bases per the plan: CamemBERT (canonical French) and XLM-R (already 3-class).
    "camembert-haca": {
        "hub_id":            "almanach/camembert-base",
        "train_lang":        "francais-haca",
        "eval_langs":        [],
        "lr":                2e-5,
        "epochs":            4,          # ~4.5k rows now (was 8 for the 143-row set)
        "batch_size":        16,
        "use_class_weights": True,
        "focal_gamma":       2.0,
    },
    "xlm-r-haca": {
        "hub_id":            "cardiffnlp/twitter-xlm-roberta-base-sentiment",
        "train_lang":        "francais-haca",
        "eval_langs":        [],
        "lr":                2e-5,
        "epochs":            4,          # ~4.5k rows now (was 8 for the 143-row set)
        "batch_size":        16,
        "use_class_weights": True,
        "focal_gamma":       2.0,
    },
    # Permissive-licence alternative for HACA production (DarijaBERT).
    "darijabert-haca": {
        "hub_id":            "SI2M-Lab/DarijaBERT",
        "train_lang":        "haca",
        "eval_langs":        ["domaine_reel_v2", "darija_ar"],
        "lr":                2e-5,
        "epochs":            4,
        "batch_size":        16,
        "use_class_weights": True,
        "focal_gamma":       2.0,
        "pos_oversample":    3,
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


def load_train_split_haca(pos_oversample: int = 1):
    """MAC training data + HACA-Sent v3 (cleaned SRT pool + synthetic), joint split.

    HACA-Sent is already de-leaked against domaine_reel_v2 (build_haca_train.py); we
    re-exclude by text as a belt-and-braces guard.  In-domain `pos` rows are optionally
    repeated `pos_oversample` times in the TRAIN split to amplify the rare class.
    """
    import importlib, sys as _sys
    _sys.path.insert(0, os.path.dirname(__file__))
    bts = importlib.import_module("build_test_sets")
    importlib.reload(bts)

    mac_df  = bts.load_mac()
    test_df = pd.read_csv(os.path.join(DATA_DIR, "darija_ar.csv"))
    mac_df  = mac_df[~mac_df["text"].isin(set(test_df["text"]))].reset_index(drop=True)

    if not os.path.exists(HACA_TRAIN):
        raise FileNotFoundError(
            f"HACA training data not found: {HACA_TRAIN}\n"
            "Run:  python src/build_haca_train.py")
    haca = pd.read_csv(HACA_TRAIN)
    haca = haca[haca["label"].isin(LABEL2ID)][["text", "label"]].reset_index(drop=True)
    frozen_v2 = pd.read_csv(os.path.join(DATA_DIR, "domaine_reel_v2.csv"))
    haca = haca[~haca["text"].isin(set(frozen_v2["text"]))].reset_index(drop=True)

    mac_tr, mac_va = train_test_split(mac_df, test_size=0.1, stratify=mac_df["label"], random_state=SEED)
    hc_tr,  hc_va  = train_test_split(haca,   test_size=0.2, stratify=haca["label"],   random_state=SEED)

    if pos_oversample and pos_oversample > 1:
        extra = pd.concat([hc_tr[hc_tr["label"] == "pos"]] * (pos_oversample - 1), ignore_index=True)
        hc_tr = pd.concat([hc_tr, extra], ignore_index=True)

    train_df = pd.concat([mac_tr, hc_tr], ignore_index=True).sample(frac=1, random_state=SEED)
    val_df   = pd.concat([mac_va, hc_va], ignore_index=True).sample(frac=1, random_state=SEED)

    print(f"  haca train: {len(train_df)} (MAC={len(mac_tr)}, haca={len(hc_tr)} incl. x{pos_oversample} pos)")
    print(f"  haca val  : {len(val_df)}   (MAC={len(mac_va)}, haca={len(hc_va)})")
    print(f"  train dist: {dict(train_df['label'].value_counts())}")
    return train_df.reset_index(drop=True), val_df.reset_index(drop=True)


def load_train_split_haca_only(pos_oversample: int = 1):
    """In-domain ONLY: HACA-Sent v3 broadcast data, no MAC in the training signal.

    Removing MAC matters because MAC is 54% positive *emotional* tweets — mixing it in
    (a) makes class weights treat `pos` as abundant and down-weight it, and (b) teaches the
    wrong `pos` concept (emojis/praise) instead of broadcast content-valence (factual good
    news).  Validation is augmented with a small MAC slice only for a stable macro-F1 signal.
    """
    import importlib, sys as _sys
    _sys.path.insert(0, os.path.dirname(__file__))
    bts = importlib.import_module("build_test_sets")
    importlib.reload(bts)

    haca = pd.read_csv(HACA_TRAIN)
    haca = haca[haca["label"].isin(LABEL2ID)][["text", "label"]].reset_index(drop=True)
    frozen_v2 = pd.read_csv(os.path.join(DATA_DIR, "domaine_reel_v2.csv"))
    haca = haca[~haca["text"].isin(set(frozen_v2["text"]))].reset_index(drop=True)

    hc_tr, hc_va = train_test_split(haca, test_size=0.2, stratify=haca["label"], random_state=SEED)
    if pos_oversample and pos_oversample > 1:
        extra = pd.concat([hc_tr[hc_tr["label"] == "pos"]] * (pos_oversample - 1), ignore_index=True)
        hc_tr = pd.concat([hc_tr, extra], ignore_index=True)

    mac_df  = bts.load_mac()
    test_df = pd.read_csv(os.path.join(DATA_DIR, "darija_ar.csv"))
    mac_df  = mac_df[~mac_df["text"].isin(set(test_df["text"]))].reset_index(drop=True)
    _, mac_va = train_test_split(mac_df, test_size=0.1, stratify=mac_df["label"], random_state=SEED)

    train_df = hc_tr.sample(frac=1, random_state=SEED)
    val_df   = pd.concat([hc_va, mac_va], ignore_index=True).sample(frac=1, random_state=SEED)
    print(f"  haca-only train: {len(train_df)} (incl. x{pos_oversample} pos)  val: {len(val_df)}")
    print(f"  train dist: {dict(train_df['label'].value_counts())}")
    return train_df.reset_index(drop=True), val_df.reset_index(drop=True)


SYNTH_FR       = os.path.join(DATA_DIR, "synthetic_haca_fr.csv")        # 143 curated clean rows
SYNTH_FR_LARGE = os.path.join(DATA_DIR, "synthetic_haca_fr_large.csv")  # thousands, templated + ASR-noised
FR_GOLD        = os.path.join(DATA_DIR, "francais_haca_gold.csv")


def load_train_split_haca_fr():
    """French HACA-Sent: hand-authored synthetic French broadcast utterances only.

    There is no real French training pool (the only real French SRT is the frozen gold), so
    the synthetic set IS the training data. Uses the LARGE generated set (templated + ASR-noise
    augmented, src/synthetic_haca_fr_large.py) when present, plus the 143 curated clean rows as
    high-quality anchors. Stratified 80/20 split; the frozen gold is excluded by text (guard).
    """
    frames = [pd.read_csv(p) for p in (SYNTH_FR_LARGE, SYNTH_FR) if os.path.exists(p)]
    if not frames:
        raise FileNotFoundError(
            "No French synthetic data found. Run:\n"
            "  python src/synthetic_haca_fr_large.py   (and/or src/synthetic_haca_fr.py)")
    df = pd.concat([f[["text", "label"]] for f in frames], ignore_index=True)
    df = df[df["label"].isin(LABEL2ID)].drop_duplicates("text").reset_index(drop=True)

    if os.path.exists(FR_GOLD):
        gold_texts = set(pd.read_csv(FR_GOLD)["text"].astype(str))
        df = df[~df["text"].isin(gold_texts)].reset_index(drop=True)

    train_df, val_df = train_test_split(
        df, test_size=0.2, stratify=df["label"], random_state=SEED)
    print(f"  francais-haca train: {len(train_df)}  val: {len(val_df)}  (sources: "
          f"{'large+curated' if len(frames) == 2 else 'single'})")
    print(f"  train dist: {dict(train_df['label'].value_counts())}")
    return train_df.reset_index(drop=True), val_df.reset_index(drop=True)


def compute_class_weights_tensor(train_df: pd.DataFrame) -> torch.Tensor:
    """Return a FloatTensor of class weights (inverse frequency, sklearn-style)."""
    labels = train_df["label"].map(LABEL2ID).values
    classes = sorted(LABEL2ID.values())
    weights = compute_class_weight("balanced", classes=np.array(classes), y=labels)
    return torch.tensor(weights, dtype=torch.float32)


class WeightedTrainer(Trainer):
    """Trainer with per-class-weighted cross-entropy, optionally focal (gamma>0).

    Focal loss down-weights easy (well-classified) examples so the model focuses on the
    hard minority classes — here the scarce broadcast `pos`/`neg`.
    """

    def __init__(self, *args, class_weights: torch.Tensor | None = None,
                 focal_gamma: float = 0.0, **kwargs):
        super().__init__(*args, **kwargs)
        self._cw = class_weights
        self._gamma = focal_gamma or 0.0

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels  = inputs.get("labels")
        outputs = model(**inputs)
        if labels is not None and (self._cw is not None or self._gamma):
            logits = outputs.logits
            weight = self._cw.to(logits.device, dtype=logits.dtype) if self._cw is not None else None
            if self._gamma:
                ce = torch.nn.functional.cross_entropy(logits, labels, weight=weight, reduction="none")
                pt = torch.exp(-ce)
                loss = ((1.0 - pt) ** self._gamma * ce).mean()
            else:
                loss = torch.nn.functional.cross_entropy(logits, labels, weight=weight)
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
    try:
        tokenizer = AutoTokenizer.from_pretrained(tok_source)
    except Exception:   # CamemBERT fast tokenizer can fail on some tokenizers versions
        tokenizer = AutoTokenizer.from_pretrained(tok_source, use_fast=False)

    # ── Load training data ────────────────────────────────────────────────────
    if train_lang == "mixed":
        train_df, val_df = load_train_split_mixed()
    elif train_lang == "broadcast":
        train_df, val_df = load_train_split_broadcast()
    elif train_lang == "haca":
        train_df, val_df = load_train_split_haca(cfg.get("pos_oversample", 1))
    elif train_lang == "haca-only":
        train_df, val_df = load_train_split_haca_only(cfg.get("pos_oversample", 1))
    elif train_lang == "francais-haca":
        train_df, val_df = load_train_split_haca_fr()
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
        save_total_limit=1,       # keep only the best checkpoint (avoids filling Kaggle disk)
        save_only_model=True,     # don't persist optimizer/scheduler state (~2x model size)
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
        trainer_kwargs["focal_gamma"] = cfg.get("focal_gamma", 0.0)

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

    try:
        _tokenizer = _AutoTokenizer.from_pretrained(ckpt_path)
    except Exception:
        _tokenizer = _AutoTokenizer.from_pretrained(ckpt_path, use_fast=False)
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
