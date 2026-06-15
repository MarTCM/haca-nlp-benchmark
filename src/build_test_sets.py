"""
Step 2 — Build and freeze the four test sets.

Run ONCE.  After this script completes the files in data/test_sets/ are READ-ONLY.
Each file has schema [text, label] with label in {neg, neu, pos}.

Usage:
    python src/build_test_sets.py

Expected raw data layout (download manually before running):
    data/raw/MAC/        — MAC dataset (LeMGarouani/MAC on GitHub)
    data/raw/ASTD/       — ASTD dataset (mahmoudnabil/ASTD on GitHub)
    data/raw/MYC/        — MYC dataset (MouadJb/MYC on GitHub)
    Allociné is fetched automatically via HuggingFace datasets.
"""

import os
import sys
import glob
import hashlib
import re
import stat

import pandas as pd
import numpy as np
from utils import set_seeds, SEED

set_seeds()

RAW_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "raw")
OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "test_sets")
os.makedirs(OUT_DIR, exist_ok=True)

N_SAMPLE = 1000
ARABIC_RE = re.compile(r"[؀-ۿ]")


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def freeze(path: str) -> None:
    """Make file read-only."""
    os.chmod(path, stat.S_IREAD | stat.S_IRGRP | stat.S_IROTH)


def sample_and_save(df: pd.DataFrame, lang: str) -> None:
    assert {"text", "label"} <= set(df.columns), f"Expected [text,label], got {list(df.columns)}"
    df = df[["text", "label"]].dropna()
    if len(df) > N_SAMPLE:
        # stratify when every class has >= 2 rows
        from sklearn.model_selection import train_test_split
        try:
            _, df = train_test_split(
                df, test_size=N_SAMPLE, stratify=df["label"], random_state=SEED
            )
        except ValueError:
            df = df.sample(N_SAMPLE, random_state=SEED)
    out = os.path.join(OUT_DIR, f"{lang}.csv")
    df.to_csv(out, index=False)
    freeze(out)
    h = sha256_file(out)
    dist = df["label"].value_counts().to_dict()
    print(f"  [{lang}] n={len(df)}  sha256={h[:16]}…  dist={dist}")


# ── 1. darija_ar — MAC ────────────────────────────────────────────────────
# Columns verified: tweets (text), type (label: positive/negative/neutral/mixed),
#                   class (dialect type: standard/dialectal — NOT the label)
def load_mac() -> pd.DataFrame:
    path = os.path.join(RAW_DIR, "MAC", "MAC corpus.csv")
    if not os.path.exists(path):
        print(
            "\n[BLOCKED] MAC dataset not found at data/raw/MAC/MAC corpus.csv\n"
            "Download from https://github.com/LeMGarouani/MAC\n"
        )
        sys.exit(1)

    df = pd.read_csv(path, on_bad_lines="skip")
    df = df.rename(columns={"tweets": "text", "type": "label"})
    df["label"] = df["label"].astype(str).str.strip().str.lower()
    label_map = {
        "positive": "pos",
        "negative": "neg",
        "neutral":  "neu",
        "mixed":    None,   # dropped per plan
    }
    df["label"] = df["label"].map(label_map)
    df = df.dropna(subset=["label"])
    return df[["text", "label"]]


# ── 2. francais — Allociné ────────────────────────────────────────────────
# Hub id verified: tblard/allocine  (original 'allocine' no longer resolves)
# Features: review (text), label ClassLabel(names=['neg','pos']) — BINARY
def load_allocine() -> pd.DataFrame:
    from datasets import load_dataset
    ds = load_dataset("tblard/allocine", split="test")
    df = ds.to_pandas()
    # ClassLabel 0=neg, 1=pos; already string after int2str
    df["label"] = df["label"].map({0: "neg", 1: "pos"})
    df = df.rename(columns={"review": "text"})
    return df[["text", "label"]]


# ── 3. msa — ASTD ─────────────────────────────────────────────────────────
# Format verified: tab-separated, no header, columns = [text, label]
# Labels: POS, NEG, OBJ (objective — dropped), MIX (mixed — dropped)
def load_astd() -> pd.DataFrame:
    path = os.path.join(RAW_DIR, "ASTD", "Tweets.txt")
    if not os.path.exists(path):
        print(
            "\n[BLOCKED] ASTD dataset not found at data/raw/ASTD/Tweets.txt\n"
            "Download data/Tweets.txt from https://github.com/mahmoudnabil/ASTD\n"
        )
        sys.exit(1)

    df = pd.read_csv(path, sep="\t", header=None, names=["text", "label"],
                     on_bad_lines="skip", encoding="utf-8")
    df["label"] = df["label"].astype(str).str.strip().str.upper()
    label_map = {
        "POS": "pos",
        "NEG": "neg",
        "OBJ": None,   # objective — dropped per plan
        "MIX": None,   # mixed — dropped per plan
    }
    df["label"] = df["label"].map(label_map)
    df = df.dropna(subset=["label"])
    return df[["text", "label"]]


# ── 4. arabizi — MYC (Latin-script rows only) ─────────────────────────────
# Format verified: UTF-16, columns = [sentence, polarity], polarity: 1=pos, -1=neg (BINARY)
# MYC is half Arabic-script / half Latin — filter to Latin rows only.
def load_myc() -> pd.DataFrame:
    path = os.path.join(RAW_DIR, "MYC", "DATA_CLEANED.csv")
    if not os.path.exists(path):
        print(
            "\n[BLOCKED] MYC dataset not found at data/raw/MYC/DATA_CLEANED.csv\n"
            "Download DATA_CLEANED.csv from https://github.com/MouadJb/MYC\n"
        )
        sys.exit(1)

    df = pd.read_csv(path, encoding="utf-16")
    df = df.rename(columns={"sentence": "text", "polarity": "label"})
    label_map = {1: "pos", -1: "neg"}
    df["label"] = df["label"].map(label_map)
    df = df.dropna(subset=["label"])

    # Filter to Latin-script rows only (no Arabic chars) — this is the arabizi set
    df = df[~df["text"].astype(str).apply(lambda t: bool(ARABIC_RE.search(t)))]
    return df[["text", "label"]]


# ── Main ──────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("Building test sets (seed=42, n≤1000 per language)…\n")

    print("darija_ar (MAC):")
    sample_and_save(load_mac(), "darija_ar")

    print("francais (Allociné):")
    sample_and_save(load_allocine(), "francais")

    print("msa (ASTD):")
    sample_and_save(load_astd(), "msa")

    print("arabizi (MYC-Latin):")
    sample_and_save(load_myc(), "arabizi")

    print("\nAll test sets built and frozen (read-only).")
    print("Next: run src/run_models.py to evaluate the ready-made models.")
