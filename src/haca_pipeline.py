"""
HACA tonality pipeline — self-hosted, segment-level, robust to noisy ASR.

Deployable inference tool (NOT a benchmark). Takes an SRT broadcast transcript and produces a
programme- and segment-level *tonality* report with confidence and human-review flags.

Design (matches the deployment constraints: on-prem, segment-level output, SRTs as-is):
  1. parse + segment the SRT (reuse srt_utils / build_haca_pool segmentation);
  2. QUALITY GATE (src/asr_quality.py): garbled ASR is *excluded from scoring*, not mislabelled;
  3. classify clean utterances with a self-hosted fine-tuned encoder + calibrated thresholds;
  4. AGGREGATE per sliding window (segment) and over the whole file (programme): pooled class
     distribution, dominant tone, mean confidence, coverage;
  5. FLAG for human review when coverage is low / no clear majority / low confidence.

Per-utterance accuracy caps ~0.5 on this content, but segment-level aggregation + abstention is
reliable and is what a regulator consumes. See report/FINDINGS.md §8.

Usage:
    # real (on a box with the checkpoint + GPU):
    python src/haca_pipeline.py --srt data/raw/srt/8.srt --model marbertv2-haca
    python src/haca_pipeline.py --srt-dir data/raw/srt --model marbertv2-haca --out-dir tonality/
    # plumbing test without a model:
    python src/haca_pipeline.py --srt data/raw/srt/8.srt --stub
"""

import argparse
import glob
import json
import os
import re
import sys
from collections import Counter

sys.path.insert(0, os.path.dirname(__file__))
from srt_utils import detect_lang                      # noqa: E402
from asr_quality import is_clean                       # noqa: E402
from extract_utterances import load_file               # noqa: E402
from build_haca_pool import utterances_for_file        # reuse the same segmentation  # noqa: E402

CLASSES = ["neg", "neu", "pos"]
WINDOW = 12          # utterances per segment (sliding window, non-overlapping)
COVERAGE_MIN = 0.40  # below this share of clean utterances -> flag for review
CONF_MIN = 0.50      # below this mean confidence -> flag
# Headline tone = dominant NON-neutral lean, gated by this floor. Neutral utterances are
# filler (definitions/transitions); a programme that is >= NONNEU_FLOOR negative is
# negative-leaning even if neutral is the plurality. Tune for the regulator's recall/precision.
NONNEU_FLOOR = 0.25
TONE_LABEL = {"neg": "negative-leaning", "neu": "neutral", "pos": "positive-leaning"}


def lean_tone(props: dict) -> str:
    """Dominant non-neutral valence gated by NONNEU_FLOOR; else neutral."""
    if props.get("neg", 0) >= NONNEU_FLOOR and props.get("neg", 0) >= props.get("pos", 0):
        return "neg"
    if props.get("pos", 0) >= NONNEU_FLOOR and props.get("pos", 0) > props.get("neg", 0):
        return "pos"
    return "neu"


# ── classifiers (pluggable) ──────────────────────────────────────────────────
def load_encoder(model_key: str):
    """Return predict_proba(texts) -> list[{neg,neu,pos}] using a fine-tuned checkpoint,
    plus calibrated thresholds if results/thresholds_<model>.json exists."""
    import torch
    from transformers import pipeline, AutoTokenizer
    from label_maps import FINETUNED_MAP, apply_map

    ckpt = f"checkpoints/{model_key}"
    tok = AutoTokenizer.from_pretrained(ckpt); tok.model_max_length = 512
    pipe = pipeline("text-classification", model=ckpt, tokenizer=tok,
                    device=0 if torch.cuda.is_available() else -1, top_k=None)

    thr = {"neg": 0.5, "neu": 0.5, "pos": 0.5}
    tpath = f"results/thresholds_{model_key}.json"
    if os.path.exists(tpath):
        thr = json.load(open(tpath)).get("thresholds", thr)

    def predict_proba(texts):
        raw = pipe(list(texts), batch_size=32, truncation=True)
        out = []
        for item in raw:
            scores = item if isinstance(item, list) else [item]
            out.append({apply_map(s["label"], FINETUNED_MAP): s["score"] for s in scores})
        return out

    return predict_proba, thr


def load_stub():
    """Keyword classifier for plumbing tests (no model). Coarse, content-valence-flavoured."""
    NEG = ["مشكل", "مشاكل", "فساد", "خساره", "خسر", "نقص", "ازمه", "فشل", "هاجر", "هجره",
           "ضعف", "غلاء", "احتجاج", "اختلاس", "رشوه", "للأسف", "حقره", "مسكين", "ضحايا", "خطير"]
    POS = ["نجاح", "نجح", "تضاعف", "استفاد", "فرصه", "انجاز", "نصر", "تحسن", "فرح", "مزيان",
           "ربح", "تطور", "نمو", "اصلاح", "فابور", "مكسب", "تاهل", "توج"]

    def predict_proba(texts):
        out = []
        for t in texts:
            t = str(t)
            n = sum(t.count(w) for w in NEG)
            p = sum(t.count(w) for w in POS)
            if n == 0 and p == 0:
                out.append({"neg": 0.2, "neu": 0.6, "pos": 0.2})
            else:
                sn, sp = n + 0.3, p + 0.3
                z = sn + sp + 0.6
                out.append({"neg": sn / z, "neu": 0.6 / z, "pos": sp / z})
        return out
    return predict_proba, {"neg": 0.5, "neu": 0.5, "pos": 0.5}


def shifted_argmax(proba: dict, thr: dict) -> str:
    return max(CLASSES, key=lambda c: proba.get(c, 0.0) - thr.get(c, 0.5))


# ── segmentation with timestamps ─────────────────────────────────────────────
def segment_srt(path: str):
    """Return list of dicts: {text, quality, clean(bool)} in reading order."""
    fmt, cues = load_file(path)
    utts = utterances_for_file(fmt, cues)
    rows = []
    for u in utts:
        u = " ".join(str(u).split()).strip()
        if len(u) < 40:
            continue
        clean, reason = is_clean(u)
        rows.append({"text": u, "clean": clean, "quality": "clean" if clean else "garbled"})
    return fmt, rows


# ── aggregation ──────────────────────────────────────────────────────────────
def aggregate(rows, thr):
    """rows already carry 'label' and 'conf' for clean utterances; aggregate them."""
    clean = [r for r in rows if r["clean"]]
    n_total, n_clean = len(rows), len(clean)
    coverage = n_clean / n_total if n_total else 0.0
    if n_clean == 0:
        return {"tone": "neu", "tone_label": "neutral", "majority": "neu",
                "distribution": {c: 0 for c in CLASSES},
                "proportions": {c: 0.0 for c in CLASSES}, "confidence": 0.0,
                "coverage": round(coverage, 3), "n_clean": 0, "n_total": n_total,
                "flag_review": True, "review_reason": "no_intelligible_content"}
    counts = Counter(r["label"] for r in clean)
    dist = {c: counts.get(c, 0) for c in CLASSES}
    props = {c: round(dist[c] / n_clean, 3) for c in CLASSES}
    majority = max(CLASSES, key=lambda c: dist[c])   # raw plurality (transparency)
    tone = lean_tone(props)                           # headline (non-neutral lean)
    conf = round(sum(r["conf"] for r in clean) / n_clean, 3)

    reasons = []
    if coverage < COVERAGE_MIN:
        reasons.append(f"low_coverage({coverage:.2f})")
    if conf < CONF_MIN:
        reasons.append(f"low_confidence({conf:.2f})")
    # borderline lean (just over the floor, and neg≈pos) -> worth a human glance
    if tone != "neu" and abs(props["neg"] - props["pos"]) < 0.10:
        reasons.append("mixed_neg_pos")
    return {"tone": tone, "tone_label": TONE_LABEL[tone], "majority": majority,
            "distribution": dist, "proportions": props,
            "confidence": conf, "coverage": round(coverage, 3),
            "n_clean": n_clean, "n_total": n_total,
            "flag_review": bool(reasons), "review_reason": ";".join(reasons) or None}


def process_file(path, predict_proba, thr):
    fmt, rows = segment_srt(path)
    clean_texts = [r["text"] for r in rows if r["clean"]]
    if clean_texts:
        probas = predict_proba(clean_texts)
        it = iter(probas)
        for r in rows:
            if r["clean"]:
                pr = next(it)
                r["label"] = shifted_argmax(pr, thr)
                r["conf"] = round(max(pr.values()), 3)

    programme = aggregate(rows, thr)
    segments = []
    for i in range(0, len(rows), WINDOW):
        seg_rows = rows[i:i + WINDOW]
        agg = aggregate(seg_rows, thr)
        agg["window"] = [i, i + len(seg_rows) - 1]
        segments.append(agg)

    return {"file": os.path.basename(path), "fmt": fmt,
            "programme": programme, "segments": segments}


def print_summary(rep):
    p = rep["programme"]
    flag = "  ⚠ REVIEW: " + (p["review_reason"] or "") if p["flag_review"] else ""
    print(f"\n{rep['file']:14s}  tone={p['tone'].upper():3s} ({p['tone_label']})  "
          f"props={p['proportions']}  conf={p['confidence']}  "
          f"coverage={p['coverage']} ({p['n_clean']}/{p['n_total']}){flag}")
    # segment timeline (compact) — uses the lean tone
    line = "  timeline: " + " ".join(
        ("·" if s["flag_review"] else {"neg": "▼", "neu": "■", "pos": "▲"}[s["tone"]])
        for s in rep["segments"])
    print(line + "    (▲pos ■neu ▼neg ·review)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--srt")
    ap.add_argument("--srt-dir")
    ap.add_argument("--model", default="marbertv2-haca")
    ap.add_argument("--stub", action="store_true", help="keyword classifier (no model) for tests")
    ap.add_argument("--out")
    ap.add_argument("--out-dir")
    ap.add_argument("--csv", help="flat segment-level CSV for a dashboard (one row per segment)")
    args = ap.parse_args()

    predict_proba, thr = load_stub() if args.stub else load_encoder(args.model)

    paths = [args.srt] if args.srt else sorted(glob.glob(os.path.join(args.srt_dir, "*.srt")))
    reports = []
    for path in paths:
        rep = process_file(path, predict_proba, thr)
        reports.append(rep)
        print_summary(rep)
        if args.out_dir:
            os.makedirs(args.out_dir, exist_ok=True)
            json.dump(rep, open(os.path.join(args.out_dir, rep["file"] + ".json"), "w"),
                      ensure_ascii=False, indent=2)
    if args.out:
        json.dump(reports, open(args.out, "w"), ensure_ascii=False, indent=2)
        print(f"\nWrote {len(reports)} report(s) -> {args.out}")

    if args.csv:
        import csv
        with open(args.csv, "w", newline="", encoding="utf-8") as fh:
            w = csv.writer(fh)
            w.writerow(["file", "level", "segment", "tone", "tone_label", "majority",
                        "p_neg", "p_neu", "p_pos", "confidence", "coverage",
                        "n_clean", "n_total", "flag_review", "reason"])
            for rep in reports:
                for level, name, a in (
                    [("programme", "-", rep["programme"])]
                    + [("segment", f"{s['window'][0]}-{s['window'][1]}", s) for s in rep["segments"]]
                ):
                    pr = a["proportions"]
                    w.writerow([rep["file"], level, name, a["tone"], a["tone_label"], a["majority"],
                                pr["neg"], pr["neu"], pr["pos"], a["confidence"], a["coverage"],
                                a["n_clean"], a["n_total"], a["flag_review"], a["review_reason"] or ""])
        print(f"Wrote dashboard CSV -> {args.csv}")


if __name__ == "__main__":
    main()
