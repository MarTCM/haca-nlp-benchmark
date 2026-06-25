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
from srt_utils import detect_lang, split_speaker, is_diarized   # noqa: E402
from asr_quality import is_clean                       # noqa: E402
from extract_utterances import load_file               # noqa: E402
from build_haca_pool import utterances_for_file, merge_cues   # reuse the same segmentation  # noqa: E402

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
# Tonality classifiers grouped by the SRT language they target. `src` is either a local
# checkpoint dir or a (locally cached) Hub id; `map` turns the model's raw labels into
# {neg,neu,pos}. The HACA models are fine-tuned 3-class encoders trained by us; the French
# entry is a general-purpose sentiment model from the Hub (NOT fine-tuned on HACA data).
def _registry():
    from label_maps import FINETUNED_MAP, DISTILCAMEMBERT_MAP, XLM_T_MAP
    return {
        # Arabe / Darija — fine-tuned 3-class checkpoints (local)
        "marbertv2-haca":     {"src": "checkpoints/marbertv2-haca",     "map": FINETUNED_MAP,
                               "lang": "arabe",    "label": "Arabe / Darija — MARBERTv2 (HACA) ★"},
        "darijabert-haca":    {"src": "checkpoints/darijabert-haca",    "map": FINETUNED_MAP,
                               "lang": "arabe",    "label": "Darija — DarijaBERT (HACA)"},
        "qarib":              {"src": "checkpoints/qarib",              "map": FINETUNED_MAP,
                               "lang": "arabe",    "label": "Arabe (MSA) — QARiB"},
        "marbertv2":          {"src": "checkpoints/marbertv2",          "map": FINETUNED_MAP,
                               "lang": "arabe",    "label": "Arabe — MARBERTv2"},
        # Arabizi (latin-script Darija)
        "darijabert-arabizi": {"src": "checkpoints/darijabert-arabizi", "map": FINETUNED_MAP,
                               "lang": "arabizi",  "label": "Arabizi — DarijaBERT"},
        # Français — HACA fine-tunes (3-class, trained on hand-authored broadcast French via
        # finetune.py). Available once checkpoints/<key>/ exists; preferred when present.
        # HACA fine-tunes on scaled+ASR-noised synthetic French. On the gold, xlm-r-haca edged
        # past off-the-shelf (0.486 vs 0.453, ~2x neutral) → recommended default; camembert-haca
        # only ties the baseline (kept selectable).
        "xlm-r-haca":         {"src": "checkpoints/xlm-r-haca",     "map": FINETUNED_MAP,
                               "lang": "francais", "label": "Français — fine-tune HACA (xlm-r-haca)"},
        "camembert-haca":     {"src": "checkpoints/camembert-haca", "map": FINETUNED_MAP,
                               "lang": "francais", "label": "Français — fine-tune HACA (camembert-haca)"},
        # Français — off-the-shelf 3-class sentiment model (neg/neu/pos), multilingual XLM-R from
        # the Hub (cached locally). Default: nothing reliably beats it on the gold (all within noise).
        "xlm-sentiment":      {"src": "cardiffnlp/twitter-xlm-roberta-base-sentiment", "map": XLM_T_MAP,
                               "lang": "francais", "label": "Français — off-the-shelf Hub (xlm-sentiment) ★"},
        # Français (alt) — Hub 5-star review model; strong on polarity but collapses neutral
        # (neutral = exactly 3★), trained on reviews not broadcast. Kept for comparison.
        "distilcamembert":    {"src": "cmarkea/distilcamembert-base-sentiment", "map": DISTILCAMEMBERT_MAP,
                               "lang": "francais", "label": "Français — distilCamemBERT (5★ reviews)"},
    }


# Default tonality model per detected language (used by the dashboard "auto" mode).
LANG_DEFAULT_MODEL = {"arabe": "marbertv2-haca", "arabizi": "darijabert-arabizi",
                      "francais": "xlm-sentiment"}


def models_for_lang(lang: str):
    """Model keys appropriate for a detected language (router output)."""
    reg = _registry()
    return [k for k, v in reg.items() if v["lang"] == lang] or list(reg)


def pick_model_for_lang(lang: str) -> str:
    """Best default tonality model for a detected language.

    French note (settled): on the 90-utterance gold, every French config lands within noise
    (0.43–0.49) — the synthetic fine-tunes (xlm-r-haca / camembert-haca), their ensemble, and
    off-the-shelf xlm-sentiment are statistically indistinguishable, and the more rigorous
    template-disjoint retrain did NOT beat off-the-shelf (see report/FINETUNING.md §6). So we
    default to the simplest, most reproducible option that nothing reliably beats: xlm-sentiment
    (no shipped checkpoint, available anywhere). The fine-tunes + `ensemble-fr` stay selectable.
    Revisit only when a larger, 2-annotator gold can actually distinguish them.
    """
    return LANG_DEFAULT_MODEL.get(lang, "marbertv2-haca")


# Probability-averaging ensembles, by key. Each member must be a load_encoder-able key.
ENSEMBLES = {"ensemble-fr": ["xlm-r-haca", "xlm-sentiment"]}


def load_ensemble(keys):
    """Average the class probabilities of several models. Robust, cheap; thresholds default 0.5."""
    members = [load_encoder(k)[0] for k in keys]

    def predict_proba(texts):
        texts = list(texts)
        preds = [fn(texts) for fn in members]
        out = []
        for i in range(len(texts)):
            agg = {}
            for p in preds:
                for c, v in p[i].items():
                    agg[c] = agg.get(c, 0.0) + v
            out.append({c: v / len(members) for c, v in agg.items()})
        return out

    return predict_proba, {"neg": 0.5, "neu": 0.5, "pos": 0.5}


def load_encoder(model_key: str):
    """Return predict_proba(texts) -> list[{neg,neu,pos}] for a registered model (or any
    checkpoints/<key> with a 3-class head), plus calibrated thresholds if
    results/thresholds_<model>.json exists."""
    if model_key in ENSEMBLES:
        return load_ensemble(ENSEMBLES[model_key])

    import torch
    from transformers import pipeline, AutoTokenizer
    from label_maps import FINETUNED_MAP, apply_map

    entry = _registry().get(model_key, {"src": f"checkpoints/{model_key}", "map": FINETUNED_MAP})
    src, lmap = entry["src"], entry["map"]
    try:
        tok = AutoTokenizer.from_pretrained(src)
    except Exception:   # CamemBERT fast tokenizer can fail on some tokenizers versions
        tok = AutoTokenizer.from_pretrained(src, use_fast=False)
    tok.model_max_length = 512
    pipe = pipeline("text-classification", model=src, tokenizer=tok,
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
            # several raw labels can map to the same class (e.g. 1+2 stars -> neg): sum them
            d = {}
            for s in scores:
                c = apply_map(s["label"], lmap)
                d[c] = d.get(c, 0.0) + s["score"]
            out.append(d)
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


def load_api_classifier(model: str = "glm-5.2", api_key: str = "",
                        url: str = "https://api.z.ai/api/paas/v4/chat/completions",
                        timeout: int = 240, include_topic: bool = False):
    """Return (predict_proba, thr) for a chat-completions API hosted classifier.

    Supports any OpenAI-compatible endpoint (Z.ai, OpenAI, Together, Groq, etc.).

    When *include_topic* is True the first call to predict_proba also extracts the
    programme topic from a representative sample and stores it as
    ``predict_proba.topic``, saving a separate topic-detection call.
    """
    import re
    import requests

    # Label aliases in multiple languages (mirrors topic_detect.snap_category).
    _SENTIMENT_ALIASES = {
        "neg": ["negatif", "negative", "négatif", "neg", "سلبي", "سالب"],
        "neu": ["neutre", "neutral", "neu", "محايد", "حيادي"],
        "pos": ["positif", "positive", "pos", "إيجابي", "موجب"],
    }

    def _snap_sentiment(raw: str) -> str | None:
        a = raw.strip().lower()
        for label, keys in _SENTIMENT_ALIASES.items():
            if any(k in a for k in keys):
                return label
        return None

    SENTIMENT_PROMPT = (
        "Tu es un assistant spécialisé dans l'analyse de la tonalité des contenus "
        "broadcast médiatiques. Analyse le texte suivant et classe-le comme "
        "'positif', 'neutre', ou 'negatif'. "
        "Réponds uniquement par un seul mot : positif, neutre, ou negatif.\n\n"
        "Texte : {text}\n"
        "Classification :"
    )

    def _call(prompt: str, max_tokens: int = 512) -> str:
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.0,
            "max_tokens": max_tokens,
        }
        r = requests.post(url, headers=headers, json=payload, timeout=timeout)
        r.raise_for_status()
        raw = r.json()["choices"][0]["message"]["content"] or ""
        # Strip possible think/reasoning tags from models that output them.
        raw = re.sub(r"<think>.*?</think>", " ", raw, flags=re.S)
        raw = re.sub(r"<think>.*$", " ", raw, flags=re.S)
        return raw.strip()

    def predict_proba(texts):
        texts_list = list(texts)
        n = len(texts_list)

        if include_topic and texts_list:
            import topic_detect as td
            head = texts_list[:4]
            rest = texts_list[4:]
            step = max(1, len(rest) // 8)
            sample = " ".join(head + rest[::step][:8])[:1500]
            topic_prompt = td.TOPIC_PROMPT.format(text=sample)
            try:
                resp = _call(topic_prompt)
                lines = [ln.strip(" .،؛:-\"'*`") for ln in resp.splitlines()]
                lines = [ln for ln in lines if ln]
                predict_proba.topic = (
                    td.snap_category(lines[0]) if lines else "غير محدد")
            except Exception:
                predict_proba.topic = None
        else:
            predict_proba.topic = None

        if n == 0:
            predict_proba.fallback_rate = 0.0
            predict_proba.first_raw = None
            return []

        # Single batch request — all utterances in one prompt, ask for one label
        # per line in order.
        delim = "\n---\n"
        body = delim.join(f"[{i}] {t}" for i, t in enumerate(texts_list))
        prompt = (
            "Analyse la tonalité de chaque énoncé ci-dessous (séparés par ---).\n"
            "Réponds avec un seul mot par énoncé dans le même ordre, un par ligne.\n"
            "Mots autorisés : positif, neutre, negatif.\n"
            "Exemple :\n"
            "positif\nneutre\nnegatif\n\n"
            f"{body}\n\n"
            "Réponse (un mot par ligne) :"
        )

        fallback_count = 0
        first_raw = None
        out = []
        try:
            raw = _call(prompt, max_tokens=min(32 * n, 2048))
            lines = raw.strip().splitlines()
            for line in lines:
                ln = line.strip().lower()
                ln = ln.strip(" .،؛:-\"'*`[]()")
                ln = re.sub(r"^\d+[.)\]\s]*", "", ln).strip()
                if not ln:
                    continue
                label = _snap_sentiment(ln)
                if label is not None:
                    d = {"neg": 0.025, "neu": 0.025, "pos": 0.025}
                    d[label] = 0.95
                    out.append(d)
                    if len(out) >= n:
                        break

            while len(out) < n:
                fallback_count += 1
                if first_raw is None:
                    first_raw = raw[:200]
                out.append({"neg": 0.33, "neu": 0.34, "pos": 0.33})
        except requests.RequestException as e:
            fallback_count = n
            first_raw = f"(exception) {e}"
            out = [{"neg": 0.33, "neu": 0.34, "pos": 0.33}] * n

        predict_proba.fallback_rate = fallback_count / n
        predict_proba.first_raw = first_raw
        return out

    predict_proba.topic = None
    predict_proba.fallback_rate = 0.0
    predict_proba.first_raw = None
    return predict_proba, {"neg": 0.5, "neu": 0.5, "pos": 0.5}


def shifted_argmax(proba: dict, thr: dict) -> str:
    return max(CLASSES, key=lambda c: proba.get(c, 0.0) - thr.get(c, 0.5))


# ── segmentation with timestamps ─────────────────────────────────────────────
def segment_srt(path: str):
    """Return list of dicts: {text, quality, clean(bool)} in reading order."""
    fmt, cues = load_file(path)
    # Strip any diarization speaker tag ([SPEAKER_XX]) from each cue so it never
    # reaches the segmentation or the classifier; per-speaker analysis handles the
    # tags separately (see segment_srt_by_speaker).
    for c in cues:
        _, c["text"] = split_speaker(c.get("text", ""))
    utts = utterances_for_file(fmt, cues)
    rows = []
    for u in utts:
        u = " ".join(str(u).split()).strip()
        if len(u) < 40:
            continue
        clean, reason = is_clean(u)
        rows.append({"text": u, "clean": clean, "quality": "clean" if clean else "garbled"})
    return fmt, rows


def segment_srt_by_speaker(path: str):
    """Group utterances by diarization speaker.

    Returns ``(fmt, diarized, by_speaker)`` where ``by_speaker`` maps each speaker
    label (``SPEAKER_00`` …) to a list of rows ``{text, clean, quality}`` — the
    same row shape as :func:`segment_srt`. Only *consecutive* cues from the same
    speaker are merged into an utterance, so one speaker's words are never pooled
    with the next person's. When the SRT is not diarized, ``diarized`` is ``False``
    and ``by_speaker`` is empty.
    """
    fmt, cues = load_file(path)
    diarized = is_diarized(cues)
    if not diarized:
        return fmt, False, {}

    # (speaker, text) in reading order; untagged cues fall under "SPEAKER_NA".
    tagged = []
    for c in cues:
        spk, txt = split_speaker(c.get("text", ""))
        tagged.append((spk or "SPEAKER_NA", txt))

    by_speaker: Dict[str, list] = {}
    i = 0
    while i < len(tagged):
        spk = tagged[i][0]
        run = []
        while i < len(tagged) and tagged[i][0] == spk:
            run.append(tagged[i][1])
            i += 1
        for u in merge_cues(run):
            u = " ".join(str(u).split()).strip()
            if len(u) < 40:
                continue
            clean, _ = is_clean(u)
            by_speaker.setdefault(spk, []).append(
                {"text": u, "clean": clean, "quality": "clean" if clean else "garbled"})
    return fmt, True, by_speaker


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


def _speaker_breakdown(path, predict_proba, thr):
    """Per-speaker tonality for a diarized SRT.

    Returns ``(diarized, speakers)`` where ``speakers`` maps each speaker to its
    aggregate report (plus ``n_utterances``), sorted by clean-utterance count.
    Every speaker's clean utterances are classified in a single batched call.
    """
    _, diarized, by_speaker = segment_srt_by_speaker(path)
    if not diarized:
        return False, {}

    # Classify all speakers' clean utterances in one pass, then map back.
    flat = [(spk, idx)
            for spk, rows in by_speaker.items()
            for idx, r in enumerate(rows) if r["clean"]]
    if flat:
        probas = predict_proba([by_speaker[spk][idx]["text"] for spk, idx in flat])
        for (spk, idx), pr in zip(flat, probas):
            r = by_speaker[spk][idx]
            r["label"] = shifted_argmax(pr, thr)
            r["conf"] = round(max(pr.values()), 3)

    speakers = {}
    for spk, rows in by_speaker.items():
        agg = aggregate(rows, thr)
        agg["n_utterances"] = len(rows)
        speakers[spk] = agg
    speakers = dict(sorted(speakers.items(),
                           key=lambda kv: kv[1]["n_clean"], reverse=True))
    return True, speakers


def process_file(path, predict_proba, thr, by_speaker=False):
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

    report = {"file": os.path.basename(path), "fmt": fmt,
              "programme": programme, "segments": segments}

    if by_speaker:
        report["diarized"], report["speakers"] = _speaker_breakdown(
            path, predict_proba, thr)

    return report


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
