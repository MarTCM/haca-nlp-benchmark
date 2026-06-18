"""
Presentation plots from a haca_pipeline.py tonality report (JSON).

Produces:
  * <out>/overview.png  — stacked bar per programme (neg/neu/pos proportions) + tone verdict;
  * <out>/<file>_timeline.png — neg/pos share per segment over time, for each neg/pos-leaning
    programme (shows WHERE the valence concentrates).

Usage:
    python src/tonality_plot.py --json tonality.json --out-dir tonality_plots
"""

import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt          # noqa: E402
from matplotlib.patches import Patch     # noqa: E402

COL = {"neg": "#d9534f", "neu": "#bdbdbd", "pos": "#5cb85c"}
FLOOR = 0.25


def overview(reports, path):
    reports = sorted(reports, key=lambda r: r["programme"]["proportions"]["neg"], reverse=True)
    fig, ax = plt.subplots(figsize=(9, 0.5 * len(reports) + 1.5))
    for i, r in enumerate(reports):
        p = r["programme"]["proportions"]
        left = 0.0
        for c in ("neg", "neu", "pos"):
            ax.barh(i, p[c], left=left, color=COL[c], edgecolor="white")
            left += p[c]
        tone = r["programme"]["tone"]
        cov = r["programme"]["coverage"]
        ax.text(1.02, i, tone.upper(), va="center", fontsize=9,
                color=COL[tone], fontweight="bold")
        ax.text(1.12, i, f"cov {cov:.0%}", va="center", fontsize=7, color="#666")
    ax.set_yticks(range(len(reports)))
    ax.set_yticklabels([r["file"] for r in reports])
    ax.set_xlim(0, 1); ax.set_xlabel("part des énoncés (proportion of utterances)")
    ax.set_title("Tonalité HACA par émission  (▼ negatif · neutre · positif ▲)")
    ax.legend(handles=[Patch(color=COL[c], label=l) for c, l in
                       [("neg", "négatif"), ("neu", "neutre"), ("pos", "positif")]],
              loc="lower right", fontsize=8)
    plt.tight_layout()
    plt.savefig(path, dpi=130, bbox_inches="tight")
    plt.close()


def timeline(report, path):
    segs = report["segments"]
    x = list(range(len(segs)))
    neg = [s["proportions"]["neg"] for s in segs]
    pos = [s["proportions"]["pos"] for s in segs]
    plt.figure(figsize=(8, 3))
    plt.plot(x, neg, color=COL["neg"], marker="o", ms=4, label="négatif")
    plt.plot(x, pos, color=COL["pos"], marker="o", ms=4, label="positif")
    plt.axhline(FLOOR, ls="--", color="#999", lw=1, label=f"seuil {FLOOR}")
    plt.ylim(0, 1); plt.xlabel("segment"); plt.ylabel("part")
    plt.title(f"{report['file']} — tonalité au fil des segments "
              f"(verdict: {report['programme']['tone'].upper()})")
    plt.legend(fontsize=8); plt.tight_layout()
    plt.savefig(path, dpi=130, bbox_inches="tight")
    plt.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", default="tonality.json")
    ap.add_argument("--out-dir", default="tonality_plots")
    args = ap.parse_args()

    reports = json.load(open(args.json))
    if not reports:
        print("empty report — nothing to plot"); return
    os.makedirs(args.out_dir, exist_ok=True)

    overview(reports, os.path.join(args.out_dir, "overview.png"))
    n = 1
    for r in reports:
        if r["programme"]["tone"] != "neu":
            timeline(r, os.path.join(args.out_dir, f"{r['file']}_timeline.png"))
            n += 1
    print(f"Wrote {n} plot(s) -> {args.out_dir}/ (overview.png + timelines for non-neutral programmes)")


if __name__ == "__main__":
    main()
