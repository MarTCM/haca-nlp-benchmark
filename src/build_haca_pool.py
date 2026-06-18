"""
Stage 1 — Quality-aware re-extraction of the HACA SRT corpus.

Produces a larger, cleaner in-domain pool than the original extract_utterances.py by:
  * window-splitting the long YouTube-format paragraphs (~530 chars, no punctuation) into
    coherent ~180-char utterances (the main lever for "more data" without new files);
  * merging short true-SRT cues into ~150-char utterances (reconstructs sentences and gives
    the garble filter enough context);
  * tagging each utterance clean|garbled via src/asr_quality.py;
  * dropping any utterance that shares a word-5-gram with the frozen gold test
    (domaine_reel_v2 / domaine_reel) — prevents content-level train/test leakage since the
    test set is drawn from these same files with different segment boundaries.

Output: data/test_sets/haca_pool_v3.csv  (unlabeled; label column empty)

Usage:
    python src/build_haca_pool.py
"""

import csv
import glob
import os
import re
import sys

sys.path.insert(0, os.path.dirname(__file__))
from extract_utterances import load_file          # reuse SRT + YouTube parsers
from srt_utils import detect_lang
from asr_quality import is_clean

SRT_DIR   = "data/raw/srt"
OUT_PATH  = "data/test_sets/haca_pool_v3.csv"
TEST_SETS = [                                       # exclude content overlapping these
    "data/test_sets/domaine_reel_v2.csv",
    "data/test_sets/domaine_reel.csv",
]

MIN_CHARS = 45
YT_MAX_CHARS  = 200     # window size for splitting long YouTube blocks
SRT_TARGET    = 150     # accumulate merged SRT cues until ~this length
SRT_MAX       = 240
NGRAM_N       = 5       # word n-gram for leakage check

_WORD_RE = re.compile(r"[^\W\d_]+", re.UNICODE)


def _norm_tokens(text: str) -> list[str]:
    return _WORD_RE.findall(text or "")


def word_ngrams(text: str, n: int = NGRAM_N) -> set[tuple]:
    toks = _norm_tokens(text)
    return {tuple(toks[i:i + n]) for i in range(len(toks) - n + 1)} if len(toks) >= n else set()


def split_block(text: str, max_chars: int = YT_MAX_CHARS, min_chars: int = MIN_CHARS) -> list[str]:
    """Greedy word-boundary windowing of a long, punctuation-free block."""
    words = text.split()
    chunks, cur = [], ""
    for w in words:
        if cur and len(cur) + 1 + len(w) > max_chars:
            chunks.append(cur)
            cur = w
        else:
            cur = f"{cur} {w}" if cur else w
    if cur:
        chunks.append(cur)
    # fold a too-short trailing chunk back into the previous one
    if len(chunks) >= 2 and len(chunks[-1]) < min_chars:
        chunks[-2] = f"{chunks[-2]} {chunks[-1]}"
        chunks.pop()
    return chunks


def merge_cues(cue_texts: list[str], target: int = SRT_TARGET, max_chars: int = SRT_MAX) -> list[str]:
    """Merge consecutive short cues into ~target-length utterances."""
    out, cur = [], ""
    for t in cue_texts:
        t = t.strip()
        if not t:
            continue
        if cur and len(cur) + 1 + len(t) > max_chars:
            out.append(cur)
            cur = t
        else:
            cur = f"{cur} {t}" if cur else t
            if len(cur) >= target:
                out.append(cur)
                cur = ""
    if cur:
        out.append(cur)
    return out


def utterances_for_file(fmt: str, cues: list[dict]) -> list[str]:
    if fmt == "youtube":
        utts = []
        for c in cues:
            utts.extend(split_block(c["text"]))
        return utts
    # true SRT: merge adjacent cues
    return merge_cues([c["text"] for c in cues])


def load_test_ngrams() -> set[tuple]:
    import pandas as pd
    grams: set[tuple] = set()
    for path in TEST_SETS:
        if os.path.exists(path):
            for t in pd.read_csv(path)["text"].astype(str):
                grams |= word_ngrams(t)
    return grams


def main():
    test_grams = load_test_ngrams()
    print(f"Loaded {len(test_grams)} test n-grams for leakage filtering")

    rows = []
    seen_texts: set[str] = set()
    stats = {}

    for path in sorted(glob.glob(os.path.join(SRT_DIR, "*.srt"))):
        fname = os.path.basename(path)
        fmt, cues = load_file(path)
        utts = utterances_for_file(fmt, cues)

        kept_clean = kept_garbled = dropped_short = dropped_leak = dropped_dup = 0
        for i, utt in enumerate(utts, start=1):
            utt = " ".join(utt.split()).strip()
            if len(utt) < MIN_CHARS:
                dropped_short += 1
                continue
            if utt in seen_texts:
                dropped_dup += 1
                continue
            if word_ngrams(utt) & test_grams:
                dropped_leak += 1
                continue
            seen_texts.add(utt)
            clean, reason = is_clean(utt)
            rows.append({
                "utterance_id": f"{fname}_{i:04d}",
                "file": fname,
                "fmt": fmt,
                "detected_lang": detect_lang(utt),
                "quality": "clean" if clean else "garbled",
                "quality_reason": reason,
                "text": utt,
                "label": "",
            })
            if clean:
                kept_clean += 1
            else:
                kept_garbled += 1
        stats[fname] = dict(fmt=fmt, clean=kept_clean, garbled=kept_garbled,
                            short=dropped_short, leak=dropped_leak, dup=dropped_dup)

    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    fields = ["utterance_id", "file", "fmt", "detected_lang", "quality",
              "quality_reason", "text", "label"]
    with open(OUT_PATH, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)

    # ── Report ────────────────────────────────────────────────────────────────
    print(f"\n{'file':12s} {'fmt':8s} {'clean':>6} {'garbled':>8} {'leak':>6} {'short':>6} {'dup':>5}")
    print("-" * 56)
    tot_clean = tot_garb = tot_leak = 0
    for fname, s in sorted(stats.items()):
        print(f"{fname:12s} {s['fmt']:8s} {s['clean']:6d} {s['garbled']:8d} "
              f"{s['leak']:6d} {s['short']:6d} {s['dup']:5d}")
        tot_clean += s["clean"]; tot_garb += s["garbled"]; tot_leak += s["leak"]
    print("-" * 56)
    print(f"{'TOTAL':12s} {'':8s} {tot_clean:6d} {tot_garb:8d} {tot_leak:6d}")
    print(f"\nPool written: {len(rows)} rows → {OUT_PATH}")
    print(f"  clean (label-ready): {tot_clean}")
    print(f"  garbled (quarantined): {tot_garb}")
    print(f"  excluded for test-leakage: {tot_leak}")


if __name__ == "__main__":
    main()
