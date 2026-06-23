"""
Stage 4 — Assemble the HACA-Sent v3 training set.

Combines:
  * data/test_sets/haca_labeled_v3.csv   — real SRT utterances, Claude-labelled (clean tier)
  * data/test_sets/synthetic_haca.csv    — Claude-authored synthetic, pos-weighted

Safeguards:
  * drops any row whose text appears in the frozen gold test (domaine_reel_v2 / domaine_reel),
  * drops rows sharing a word-5-gram with the gold test (content-level leakage),
  * de-duplicates exact text.

Output: data/test_sets/haca_train_v3.csv  (the file finetune.py consumes)

Usage:
    python src/build_haca_train.py
"""

import os
import re
from collections import Counter

import pandas as pd

LABELED   = "data/test_sets/haca_labeled_v3.csv"
SYNTH     = "data/test_sets/synthetic_haca.csv"
TEST_SETS = ["data/test_sets/domaine_reel_v2.csv", "data/test_sets/domaine_reel.csv"]
OUT_CSV   = "data/test_sets/haca_train_v3.csv"

_WORD = re.compile(r"[^\W\d_]+", re.UNICODE)
COLS  = ["utterance_id", "file", "fmt", "detected_lang", "quality", "text",
         "label", "label_source", "synthetic", "topic"]


def grams(text, n=5):
    w = _WORD.findall(str(text))
    return {tuple(w[i:i + n]) for i in range(len(w) - n + 1)} if len(w) >= n else set()


def main():
    real = pd.read_csv(LABELED)
    real["synthetic"] = False
    real["topic"] = ""

    synth = pd.read_csv(SYNTH)

    df = pd.concat([real[COLS], synth[COLS]], ignore_index=True)

    # ── leakage / dedup safeguards ──────────────────────────────────────────
    test_texts = set()
    test_grams = set()
    for f in TEST_SETS:
        if os.path.exists(f):
            for t in pd.read_csv(f)["text"].astype(str):
                test_texts.add(t)
                test_grams |= grams(t)

    before = len(df)
    df = df[~df["text"].isin(test_texts)]
    df = df[~df["text"].apply(lambda t: bool(grams(t) & test_grams))]
    df = df.drop_duplicates(subset=["text"]).reset_index(drop=True)
    dropped = before - len(df)

    df.to_csv(OUT_CSV, index=False)

    dist = Counter(df["label"])
    src  = Counter(df["label_source"])
    print(f"Wrote {len(df)} rows → {OUT_CSV}  (dropped {dropped} for leakage/dups)")
    print(f"  label distribution : {dict(dist)}")
    print(f"  by source          : {dict(src)}")
    print(f"  synthetic share    : {df['synthetic'].sum()} / {len(df)} "
          f"({df['synthetic'].mean()*100:.0f}%)")
    real_pos = ((df.label == 'pos') & (~df.synthetic)).sum()
    syn_pos  = ((df.label == 'pos') & (df.synthetic)).sum()
    print(f"  pos: {real_pos} real + {syn_pos} synthetic = {real_pos + syn_pos}")


if __name__ == "__main__":
    main()
