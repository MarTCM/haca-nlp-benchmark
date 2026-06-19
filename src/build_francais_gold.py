"""
French HACA gold test set — hand-labelled real broadcast SRT.

NO LLM API. The labels below are assigned by Claude (the assistant) reading each clean
utterance of data/raw/srt/emission_francaise.srt (a French TV "grand format" on Morocco:
the tomato industry, migrant labour, the Gen-Z protests, and French investment). Content-
valence rubric (report/ANNOTATION_RUBRIC_V3.md):
  * pos — reports something good (growth, modernization, investment, achievement);
  * neg — reports something bad (shortage, exploitation, repression, inequality, death);
  * neu — descriptive / procedural / framing, no recoverable valence.

This is the FROZEN gold for evaluating the French fine-tunes — it is real in-domain data and
is NEVER used for training (the training pool is src/synthetic_haca_fr.py only).

The labels are keyed by the utterance index produced by haca_pipeline.segment_srt(); the
script re-derives the text from the SRT so the gold stays in sync, and guards on the count.

Usage:
    python src/build_francais_gold.py     # writes data/test_sets/francais_haca_gold.csv
"""

import csv
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
import haca_pipeline as hp  # noqa: E402

SRT = "data/raw/srt/emission_francaise.srt"
OUT_CSV = "data/test_sets/francais_haca_gold.csv"

# index (in the clean-utterance order) -> hand label
LABELS = {
    0: "neu", 1: "neg", 2: "neu", 3: "neu", 4: "neu", 5: "neu", 6: "neg", 7: "neu",
    8: "neu", 9: "neu", 10: "pos", 11: "neu", 12: "pos", 13: "neu", 14: "neu", 15: "neu",
    16: "neg", 17: "neu", 18: "neg", 19: "neg", 20: "neg", 21: "neg", 22: "neg", 23: "neg",
    24: "neg", 25: "neg", 26: "neg", 27: "neg", 28: "neu", 29: "neg", 30: "neg", 31: "neg",
    32: "neg", 33: "neg", 34: "neu", 35: "neg", 36: "neg", 37: "neg", 38: "neg", 39: "neu",
    40: "neu", 41: "neg", 42: "neg", 43: "neg", 44: "neg", 45: "neg", 46: "neg", 47: "neg",
    48: "neg", 49: "neg", 50: "neg", 51: "neg", 52: "neg", 53: "neu", 54: "neu", 55: "neg",
    56: "pos", 57: "pos", 58: "neg", 59: "neg", 60: "neg", 61: "neg", 62: "neg", 63: "neg",
    64: "pos", 65: "neu", 66: "pos", 67: "neu", 68: "neu", 69: "neu", 70: "pos", 71: "pos",
    72: "pos", 73: "neu", 74: "pos", 75: "pos", 76: "pos", 77: "neu", 78: "neu", 79: "neu",
    80: "neu", 81: "neu", 82: "neu", 83: "pos", 84: "pos", 85: "pos", 86: "pos", 87: "pos",
    88: "neg", 89: "pos",
}


def main():
    _, rows = hp.segment_srt(SRT)
    clean = [r["text"] for r in rows if r["clean"]]
    assert len(clean) == len(LABELS), (
        f"clean utterance count {len(clean)} != labelled {len(LABELS)} — "
        "segmentation changed; re-check the index→label map.")
    assert clean[0].startswith("Et on envent"), "utterance 0 changed — re-check labels"

    out = []
    for i, text in enumerate(clean):
        out.append({
            "utterance_id": f"emission_francaise_{i:04d}",
            "file": "emission_francaise.srt",
            "fmt": "srt",
            "detected_lang": "francais",
            "quality": "clean",
            "text": text,
            "label": LABELS[i],
            "label_source": "claude-manual",
            "synthetic": False,
            "topic": "",
        })

    os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)
    fields = ["utterance_id", "file", "fmt", "detected_lang", "quality", "text",
              "label", "label_source", "synthetic", "topic"]
    with open(OUT_CSV, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        w.writerows(out)
    from collections import Counter
    print(f"Wrote {len(out)} gold rows → {OUT_CSV}")
    print(f"  distribution: {dict(Counter(r['label'] for r in out))}")


if __name__ == "__main__":
    main()
