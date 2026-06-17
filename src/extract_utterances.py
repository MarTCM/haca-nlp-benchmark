"""
Extract utterances from SRT files (and YouTube transcript dumps)
and export a CSV ready for manual sentiment annotation.

Usage:
    python src/extract_utterances.py
    python src/extract_utterances.py --sample 200 --out data/test_sets/domaine_reel_raw.csv
"""

import argparse
import csv
import glob
import os
import re
import sys

sys.path.insert(0, os.path.dirname(__file__))
from srt_utils import cues_to_utterances, detect_lang, parse_srt

# YouTube auto-transcript format: "(mm:ss) text" or "(hh:mm:ss) text"
_YT_BLOCK = re.compile(r"\(\d{1,2}:\d{2}(?::\d{2})?\)\s+(.+?)(?=\s*\(\d|$)", re.DOTALL)
_YT_TITLE = re.compile(r"^.+- YouTube\s*$", re.MULTILINE)
_YT_URL   = re.compile(r"^https?://", re.MULTILINE)


def _parse_youtube_transcript(path: str) -> list[dict]:
    """Extract text blocks from YouTube (mm:ss) transcript format."""
    for enc in ("utf-8", "utf-8-sig", "cp1256", "latin-1"):
        try:
            with open(path, encoding=enc) as fh:
                content = fh.read()
            break
        except (UnicodeDecodeError, LookupError):
            continue
    else:
        return []

    cues = []
    for i, m in enumerate(_YT_BLOCK.finditer(content), start=1):
        text = m.group(1).replace("\n", " ").strip()
        if text:
            cues.append({"index": i, "start": "", "end": "", "text": text})
    return cues


def load_file(path: str) -> tuple[str, list[dict]]:
    """
    Return (format_type, cues) for any supported file.
    format_type is 'srt' or 'youtube'.
    """
    # Try SRT first
    try:
        cues = parse_srt(path)
        if len(cues) >= 3:
            return ("srt", cues)
    except Exception:
        pass

    # Fall back to YouTube transcript
    yt_cues = _parse_youtube_transcript(path)
    if yt_cues:
        return ("youtube", yt_cues)

    return ("unknown", [])


def _cues_to_utterances_robust(cues: list[dict], fmt: str) -> list[str]:
    """
    For YouTube blocks: keep each block as its own utterance (they are already
    paragraph-length). For SRT: try sentence-boundary splitting; if it collapses
    everything into 1 utterance (no punctuation), fall back to per-cue splits.
    """
    if fmt == "youtube":
        # Each YouTube (mm:ss) block is already a coherent paragraph.
        return [c["text"] for c in cues if c["text"].strip()]

    utts = cues_to_utterances(cues)
    # If sentence splitting produced ≤1 result for a multi-cue file, the
    # subtitles have no sentence-ending punctuation — use per-cue splits instead.
    if len(utts) <= 1 and len(cues) > 10:
        utts = [c["text"].strip() for c in cues if c["text"].strip()]
    return utts


def extract_all(srt_dir: str) -> list[dict]:
    """
    Walk all *.srt files under srt_dir and return a flat list of utterance dicts:
        {file, fmt, utterance_id, text, detected_lang}
    """
    rows = []
    paths = sorted(glob.glob(os.path.join(srt_dir, "*.srt")))
    if not paths:
        print(f"No .srt files found in {srt_dir}", file=sys.stderr)
        return rows

    for path in paths:
        fname = os.path.basename(path)
        fmt, cues = load_file(path)
        if not cues:
            print(f"  [skip] {fname} — could not parse", file=sys.stderr)
            continue

        utterances = _cues_to_utterances_robust(cues, fmt)
        n = len(utterances)
        print(f"  {fname:20s}  fmt={fmt:7s}  cues={len(cues):4d}  utterances={n:4d}")

        for i, utt in enumerate(utterances, start=1):
            utt = utt.strip()
            if len(utt) < 40:    # skip noise/very short fragments
                continue
            if len(utt) > 400:   # truncate very long YouTube blocks at word boundary
                utt = utt[:400].rsplit(" ", 1)[0] + " ..."
            rows.append({
                "file": fname,
                "fmt": fmt,
                "utterance_id": f"{fname}_{i:04d}",
                "text": utt,
                "detected_lang": detect_lang(utt),
                "label": "",  # to be filled by annotator
            })

    return rows


def stratified_sample(rows: list[dict], n: int) -> list[dict]:
    """
    Draw up to n rows, evenly distributed across all files.
    Every file contributes floor(n / num_files) utterances, remainder filled
    round-robin, so no single file dominates.
    """
    from collections import defaultdict
    import math

    by_file: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_file[r["file"]].append(r)

    num_files = len(by_file)
    base = n // num_files          # each file gets at least this many
    extra = n - base * num_files   # distribute remainder round-robin

    result = []
    for i, (fname, items) in enumerate(sorted(by_file.items())):
        quota = base + (1 if i < extra else 0)
        quota = min(quota, len(items))
        step = max(1, len(items) // quota)
        result.extend(items[::step][:quota])

    return result[:n]


def write_csv(rows: list[dict], out_path: str) -> None:
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fieldnames = ["utterance_id", "file", "fmt", "detected_lang", "text", "label"]
    with open(out_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row[k] for k in fieldnames})
    print(f"\nWrote {len(rows)} rows → {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--srt-dir", default="data/raw/srt")
    parser.add_argument("--sample", type=int, default=200,
                        help="Number of utterances to sample (0 = keep all)")
    parser.add_argument("--out", default="data/test_sets/domaine_reel_raw.csv")
    args = parser.parse_args()

    print(f"Scanning {args.srt_dir} ...")
    rows = extract_all(args.srt_dir)
    print(f"\nTotal utterances extracted: {len(rows)}")

    # Language breakdown
    from collections import Counter
    lang_counts = Counter(r["detected_lang"] for r in rows)
    for lang, cnt in sorted(lang_counts.items()):
        print(f"  {lang:12s}: {cnt}")

    if args.sample and len(rows) > args.sample:
        rows = stratified_sample(rows, args.sample)
        print(f"\nStratified sample: {len(rows)} utterances")
        lang_counts2 = Counter(r["detected_lang"] for r in rows)
        for lang, cnt in sorted(lang_counts2.items()):
            print(f"  {lang:12s}: {cnt}")

    write_csv(rows, args.out)


if __name__ == "__main__":
    main()
