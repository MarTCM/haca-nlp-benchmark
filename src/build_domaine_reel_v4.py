"""
Build domaine_reel_v4_balanced.csv — a BALANCED DIAGNOSTIC eval set hitting the protocol's
per-class minimums (>=100 pos / >=150 neg / >=200 neu) from the 12 SRTs available now.

Composition:
  * REAL (source=human_gold): the 194 human-annotated domaine_reel_v2 utterances — truly
    held out (never trained on, de-leaked). Gives 20 pos / 63 neg / 111 neu.
  * SYNTHETIC (source=synthetic): fresh Claude-authored utterances (src/synthetic_haca_test.py),
    distinct from the training synthetic, added to top each class up to the protocol count.

Guards: drop any synthetic row that duplicates / shares a 5-gram with the training set
(haca_train_v3) or with domaine_reel_v2; freeze with SHA-256.

⚠ NOT a gold benchmark. Synthetic positives measure style-memorisation (models were trained on
Claude-authored synthetic positives), so pos-F1 on the full set is OPTIMISTIC. Always report the
real-only subset (source=human_gold) alongside the full balanced numbers. See TEST_SET_PROTOCOL.md.

Usage:
    python src/build_domaine_reel_v4.py
"""

import hashlib
import os
import re
from collections import Counter

import pandas as pd

import synthetic_haca_test as syn

REAL_CSV  = "data/test_sets/domaine_reel_v2.csv"
TRAIN_CSV = "data/test_sets/haca_train_v3.csv"
OUT_CSV   = "data/test_sets/domaine_reel_v4_balanced.csv"
TARGET    = {"pos": 100, "neg": 150, "neu": 200}

_WORD = re.compile(r"[^\W\d_]+", re.UNICODE)
COLS  = ["utterance_id", "file", "fmt", "detected_lang", "text", "label", "source"]


def grams(t, n=5):
    w = _WORD.findall(str(t))
    return {tuple(w[i:i + n]) for i in range(len(w) - n + 1)} if len(w) >= n else set()


def main():
    real = pd.read_csv(REAL_CSV)
    real = real[["utterance_id", "file", "fmt", "detected_lang", "text", "label"]].copy()
    real["source"] = "human_gold"

    # Leak guard against REAL training content only (the held-out real test is
    # domaine_reel_v2 itself). We deliberately do NOT block overlap with the *training
    # synthetic* — two templated sentences sharing a phrase isn't real-content leakage; it
    # only makes the synthetic top-up even more optimistic (already flagged as such).
    train = pd.read_csv(TRAIN_CSV)
    # exact-dup guard: against ALL training (incl. synthetic) + the real test
    train_texts = set(train["text"].astype(str)) | set(real["text"].astype(str))
    # 5-gram guard: against REAL training only (phrase overlap with training synthetic is ok)
    train_real = train[train["label_source"] != "claude-synth"]
    train_grams = set()
    for t in train_real["text"].astype(str):
        train_grams |= grams(t)

    syn_lists = {"pos": syn.POS, "neg": syn.NEG, "neu": syn.NEU}
    real_counts = Counter(real["label"])
    rows, sid = [], 0
    for label, target in TARGET.items():
        need = target - real_counts.get(label, 0)
        pool = [t for (t, _) in syn_lists[label]]
        kept = []
        for t in pool:
            t = " ".join(str(t).split()).strip()
            if t in train_texts:                       # exact dup vs train/real
                continue
            if grams(t) & train_grams:                 # 5-gram leak vs train
                continue
            if t in {r["text"] for r in kept}:         # internal dup
                continue
            kept.append({"utterance_id": f"synth_test_{sid:04d}", "file": "synthetic",
                         "fmt": "synthetic", "detected_lang": "arabe", "text": t,
                         "label": label, "source": "synthetic"})
            sid += 1
            if len(kept) >= need:
                break
        if len(kept) < need:
            print(f"  [warn] {label}: only {len(kept)} clean synthetic available, need {need}")
        rows.extend(kept)

    out = pd.concat([real, pd.DataFrame(rows)[COLS]], ignore_index=True)
    out = out.drop_duplicates(subset=["text"]).reset_index(drop=True)
    out.to_csv(OUT_CSV, index=False)

    sha = hashlib.sha256(open(OUT_CSV, "rb").read()).hexdigest()
    print(f"Wrote {len(out)} rows -> {OUT_CSV}")
    print(f"  distribution : {dict(Counter(out['label']))}")
    print(f"  by source    : {dict(Counter(out['source']))}")
    print("  per class (real human_gold + synthetic):")
    for lab in ["neg", "neu", "pos"]:
        r = ((out.label == lab) & (out.source == "human_gold")).sum()
        s = ((out.label == lab) & (out.source == "synthetic")).sum()
        print(f"    {lab}: {r} real + {s} synthetic = {r + s}")
    print(f"  SHA-256      : {sha}")


if __name__ == "__main__":
    import sys
    sys.path.insert(0, os.path.dirname(__file__))
    main()
