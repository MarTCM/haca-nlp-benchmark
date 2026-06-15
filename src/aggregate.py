"""
Step 7 — Aggregate results, produce plots and weighted scoring grid.

Usage:
    python src/aggregate.py
"""

import os
import json
import glob

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns

RESULTS_DIR  = os.path.join(os.path.dirname(__file__), "..", "results")
FIGS_DIR     = os.path.join(RESULTS_DIR, "figs")
SUMMARY_CSV  = os.path.join(RESULTS_DIR, "summary.csv")
os.makedirs(FIGS_DIR, exist_ok=True)

LANGS  = ["darija_ar", "francais", "msa", "arabizi"]
MODELS = ["xlm-t", "camelbert-da", "distilcamembert",
          "darijabert", "darijabert-arabizi", "marbertv2", "qarib",
          "atlas-chat-2b", "atlas-chat-9b"]

# Weighted scoring weights (per plan)
W_PRECISION   = 0.50   # macro-F1 averaged per language
W_INTEGRATION = 0.20   # dummy = 1.0 for all (encoder) — placeholder
W_COST        = 0.20   # inverted: latency, memory, GPU need
W_COVERAGE    = 0.10   # number of languages covered / 4


def load_summary() -> pd.DataFrame:
    if not os.path.exists(SUMMARY_CSV):
        raise FileNotFoundError(f"{SUMMARY_CSV} not found. Run run_models.py first.")
    return pd.read_csv(SUMMARY_CSV)


# ── Macro-F1 heatmap ──────────────────────────────────────────────────────

def plot_heatmap(df: pd.DataFrame) -> None:
    pivot = df.pivot_table(index="model", columns="lang", values="macro_f1", aggfunc="mean")
    pivot = pivot.reindex(columns=LANGS)

    fig, ax = plt.subplots(figsize=(9, max(4, len(pivot) * 0.55 + 1)))
    sns.heatmap(
        pivot, annot=True, fmt=".3f", cmap="RdYlGn",
        vmin=0, vmax=1, linewidths=0.5, ax=ax,
    )
    ax.set_title("Macro-F1 by Model × Language")
    ax.set_xlabel("Language")
    ax.set_ylabel("Model")
    plt.tight_layout()
    out = os.path.join(FIGS_DIR, "heatmap_f1.png")
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"  Saved {out}")


# ── Cost-precision scatter ─────────────────────────────────────────────────

def plot_scatter(df: pd.DataFrame) -> None:
    agg = df.groupby("model").agg(
        mean_f1=("macro_f1", "mean"),
        mean_latency=("latency_ms", "mean"),
        n_params=("n_params", "first"),
    ).reset_index()

    agg["bubble"] = np.sqrt(agg["n_params"].clip(lower=1) / 1e6)

    fig, ax = plt.subplots(figsize=(9, 6))
    scatter = ax.scatter(
        agg["mean_latency"], agg["mean_f1"],
        s=agg["bubble"] * 10, alpha=0.7,
        c=range(len(agg)), cmap="tab10",
    )
    for _, row in agg.iterrows():
        ax.annotate(row["model"], (row["mean_latency"], row["mean_f1"]),
                    textcoords="offset points", xytext=(5, 3), fontsize=8)
    ax.set_xscale("log")
    ax.set_xlabel("Median latency (ms/utterance) — log scale")
    ax.set_ylabel("Mean macro-F1 across languages")
    ax.set_title("Cost-Precision scatter (bubble = √params in M)")
    plt.tight_layout()
    out = os.path.join(FIGS_DIR, "scatter_cost_precision.png")
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"  Saved {out}")


# ── Weighted scoring grid ─────────────────────────────────────────────────

def _minmax(series: pd.Series, invert=False) -> pd.Series:
    mn, mx = series.min(), series.max()
    if mx == mn:
        return pd.Series(0.5, index=series.index)
    norm = (series - mn) / (mx - mn)
    return 1 - norm if invert else norm


def compute_weighted_grid(df: pd.DataFrame) -> pd.DataFrame:
    agg = df.groupby("model").agg(
        mean_f1=("macro_f1", "mean"),
        mean_latency=("latency_ms", "mean"),
        mean_vram=("peak_vram_mb", "mean"),
        n_langs=("lang", "nunique"),
    ).reset_index()

    agg["s_precision"]   = _minmax(agg["mean_f1"])
    agg["s_integration"] = 1.0  # placeholder — all models expose the same API
    cost_raw = _minmax(agg["mean_latency"]) * 0.6 + _minmax(agg["mean_vram"]) * 0.4
    agg["s_cost"]       = 1 - cost_raw        # invert: lower cost = higher score
    agg["s_coverage"]   = agg["n_langs"] / 4

    agg["weighted_score"] = (
        W_PRECISION   * agg["s_precision"]   +
        W_INTEGRATION * agg["s_integration"] +
        W_COST        * agg["s_cost"]        +
        W_COVERAGE    * agg["s_coverage"]
    )
    return agg.sort_values("weighted_score", ascending=False).reset_index(drop=True)


def plot_radar(df_grid: pd.DataFrame, top_n: int = 5) -> None:
    from matplotlib.patches import FancyArrowPatch

    finalists = df_grid.head(top_n)
    categories = ["Precision", "Integration", "Cost", "Coverage"]
    N = len(categories)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(7, 7), subplot_kw=dict(polar=True))
    for _, row in finalists.iterrows():
        values = [
            row["s_precision"], row["s_integration"],
            row["s_cost"],      row["s_coverage"],
        ]
        values += values[:1]
        ax.plot(angles, values, label=row["model"])
        ax.fill(angles, values, alpha=0.1)
    ax.set_thetagrids(np.degrees(angles[:-1]), categories)
    ax.set_title(f"Radar chart — top {top_n} finalists")
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1))
    plt.tight_layout()
    out = os.path.join(FIGS_DIR, "radar_finalists.png")
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"  Saved {out}")


def print_recommendations(df_grid: pd.DataFrame, df: pd.DataFrame) -> None:
    print("\n=== Per-scenario recommendations ===")

    best_overall = df_grid.iloc[0]["model"]
    print(f"Max-precision routing   : {best_overall} (highest weighted score)")

    cheapest = df_grid.sort_values("s_cost", ascending=False).iloc[0]["model"]
    print(f"Min-cost single model   : {cheapest}")

    balanced = df_grid.sort_values("weighted_score", ascending=False).iloc[0]["model"]
    print(f"Balanced                : {balanced}")

    # Decision threshold: prefer fine-tuned encoder over LLM if darija macro-F1 >= 0.75
    # or if LLM-encoder gap < 5 F1 points
    darija_f1 = df[df["lang"] == "darija_ar"].set_index("model")["macro_f1"]
    llm_models = [m for m in darija_f1.index if "atlas" in m]
    enc_models = [m for m in darija_f1.index if "atlas" not in m]
    if llm_models and enc_models:
        best_enc_f1 = darija_f1[enc_models].max()
        best_llm_f1 = darija_f1[llm_models].max()
        gap = best_llm_f1 - best_enc_f1
        if best_enc_f1 >= 0.75 or gap < 0.05:
            print(
                f"\n  [Threshold met] Prefer fine-tuned encoder over LLM on darija "
                f"(encoder F1={best_enc_f1:.3f}, gap={gap:+.3f})"
            )
        else:
            print(
                f"\n  [Threshold NOT met] LLM advantage on darija is significant "
                f"(encoder F1={best_enc_f1:.3f}, gap={gap:+.3f}) — consider LLM."
            )


if __name__ == "__main__":
    print("Loading results…")
    df = load_summary()

    print("\n--- Macro-F1 pivot ---")
    pivot = df.pivot_table(index="model", columns="lang", values="macro_f1").round(4)
    print(pivot.to_string())

    print("\nGenerating plots…")
    plot_heatmap(df)
    plot_scatter(df)

    print("\nWeighted scoring grid:")
    grid = compute_weighted_grid(df)
    print(grid[["model", "mean_f1", "s_precision", "s_cost", "s_coverage", "weighted_score"]].to_string(index=False))

    grid.to_csv(os.path.join(RESULTS_DIR, "weighted_grid.csv"), index=False)

    plot_radar(grid)
    print_recommendations(grid, df)

    print("\nDone.")
