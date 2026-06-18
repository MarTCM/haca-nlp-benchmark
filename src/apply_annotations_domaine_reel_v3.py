"""
Re-annotation of the 194 domaine_reel_v2 utterances under Rubric v3 (content-valence),
authored by Claude — a CONSISTENCY STUDY, not a replacement gold.

Why: §8.4 of FINDINGS showed the broadcast test's positive class is the limiting factor.
Some test positives come from source files absent from the repo, and its pos/neu boundary
differs from rubric v3. This file re-labels the *same 194 texts* under rubric v3, applied
consistently (e.g. Moroccan historical victories framed positively -> pos; garbled
fragments and borderline "a service exists / a rule" items -> neu), so we can:
  1. measure how much the labels move (Cohen's kappa) — i.e. how subjective the task is;
  2. give a rubric-aligned test the v3-trained models can be re-evaluated on.

IMPORTANT — provenance: domaine_reel_v2 is HUMAN-annotated and remains the independent gold.
domaine_reel_v3 is CLAUDE-annotated (same rubric as the training labels), so evaluating
Claude-trained models on it carries an alignment bias. It is a consistency artefact + a
candidate the human should adjudicate (second annotator) before it becomes an official gold.

Usage:
    python src/apply_annotations_domaine_reel_v3.py     # writes domaine_reel_v3.csv + report
"""

import os
import pandas as pd

SRC = "data/test_sets/domaine_reel_v2.csv"
OUT = "data/test_sets/domaine_reel_v3.csv"

# utterance_id -> rubric-v3 label (content-valence), authored by Claude.
V3 = {
    # 1.srt — garbled drama: all neu (no recoverable valence)
    "1.srt_0007": "neu", "1.srt_0095": "neu", "1.srt_0135": "neu", "1.srt_0216": "neu",
    "1.srt_0296": "neu", "1.srt_0306": "neu", "1.srt_0327": "neu", "1.srt_0441": "neu",
    "1.srt_0508": "neu", "1.srt_0538": "neu", "1.srt_0560": "neu",
    # 10.srt — health: out-of-pocket/shortage/emigration/governance = neg; private-sector
    # reform increases supply + lowers price = pos; description/intro = neu
    "10.srt_0001": "neg", "10.srt_0005": "neu", "10.srt_0009": "neu", "10.srt_0013": "neu",
    "10.srt_0017": "neu", "10.srt_0021": "neu", "10.srt_0025": "neu", "10.srt_0029": "neu",
    "10.srt_0033": "neu", "10.srt_0037": "neg", "10.srt_0041": "neg", "10.srt_0045": "neg",
    "10.srt_0049": "neg", "10.srt_0053": "neg", "10.srt_0057": "neg", "10.srt_0061": "neg",
    "10.srt_0065": "pos",
    "11.srt_0059": "neu", "11.srt_0182": "neu", "11.srt_0190": "neu", "11.srt_0209": "neu",
    "12.srt_0016": "neu", "12.srt_0097": "neu", "12.srt_0109": "neu", "12.srt_0171": "neg",
    "13.srt_0069": "neu", "13.srt_0101": "neu", "13.srt_0110": "neu", "13.srt_0192": "neu",
    "13.srt_0219": "neu", "13.srt_0284": "neu", "13.srt_0313": "neu",
    "14.srt_0006": "neu", "14.srt_0050": "neu",
    # 2.srt — tax: unfairness/burden = neg; mechanism/intro = neu
    "2.srt_0001": "neu", "2.srt_0005": "neu", "2.srt_0009": "neu", "2.srt_0013": "neu",
    "2.srt_0017": "neg", "2.srt_0021": "neu", "2.srt_0025": "neu", "2.srt_0029": "neu",
    "2.srt_0033": "neu", "2.srt_0037": "neu", "2.srt_0041": "neg", "2.srt_0045": "neu",
    "2.srt_0049": "neu", "2.srt_0053": "neg", "2.srt_0057": "neg", "2.srt_0061": "neu",
    "2.srt_0065": "neu",
    # 3.srt — procurement: opportunity ("245 billion") = pos; procedure = neu
    "3.srt_0001": "pos", "3.srt_0047": "neu", "3.srt_0055": "neu", "3.srt_0059": "neu",
    "3.srt_0085": "neu", "3.srt_0089": "neu", "3.srt_0093": "neu", "3.srt_0109": "neu",
    "3.srt_0113": "neu", "3.srt_0125": "neu", "3.srt_0129": "neu", "3.srt_0133": "neu",
    "3.srt_0153": "neu", "3.srt_0157": "neu", "3.srt_0169": "neu", "3.srt_0177": "neu",
    "3.srt_0185": "neu",
    # 4.srt — stock market: "shares 7x" = pos; belittled = neg; mechanism = neu
    "4.srt_0001": "pos", "4.srt_0005": "neg", "4.srt_0009": "neu", "4.srt_0013": "neu",
    "4.srt_0025": "neu", "4.srt_0045": "neu", "4.srt_0049": "neu", "4.srt_0057": "neu",
    "4.srt_0065": "neu", "4.srt_0069": "neu", "4.srt_0081": "neu", "4.srt_0085": "neu",
    "4.srt_0101": "neu", "4.srt_0105": "neu", "4.srt_0109": "neu", "4.srt_0113": "neu",
    "4.srt_0121": "neu",
    "5.srt_0028": "neu", "5.srt_0099": "neu", "5.srt_0137": "neu", "5.srt_0201": "neu",
    # 6.srt / 7.srt / 8.srt — Sufi + wisdom/freedom sermon + slavery history: neu
    "6.srt_0110": "neu", "6.srt_0116": "neu", "6.srt_0121": "neu",
    "7.srt_0001": "neu", "7.srt_0070": "neu", "7.srt_0150": "neu", "7.srt_0153": "neu",
    "7.srt_0185": "neu", "7.srt_0209": "neu",
    "8.srt_0008": "neu", "8.srt_0083": "neu", "8.srt_0095": "neu", "8.srt_0110": "neu",
    # 9.srt — MODERN Sahara: diplomatic victories/development/recovery = pos; loss/war = neg
    "9.srt_0001": "pos", "9.srt_0005": "neu", "9.srt_0021": "neg", "9.srt_0025": "neu",
    "9.srt_0033": "neg", "9.srt_0037": "neg", "9.srt_0041": "neg", "9.srt_0053": "pos",
    "9.srt_0065": "neg", "9.srt_0069": "neu", "9.srt_0081": "neg", "9.srt_0093": "pos",
    "9.srt_0109": "pos", "9.srt_0113": "pos", "9.srt_0125": "pos", "9.srt_0137": "pos",
    "9.srt_0141": "pos",
    # d1/d2.srt — debates: govt failures/accusations = neg; party hope = pos; procedure = neu
    "d1.srt_0001": "neu", "d1.srt_0002": "neu", "d1.srt_0007": "neu", "d1.srt_0020": "neu",
    "d1.srt_0028": "neg", "d1.srt_0036": "neg", "d1.srt_0045": "neu", "d1.srt_0055": "neg",
    "d1.srt_0060": "neg", "d1.srt_0064": "neg", "d1.srt_0067": "neg", "d1.srt_0078": "neg",
    "d1.srt_0083": "neu", "d1.srt_0087": "neu", "d1.srt_0091": "neu", "d1.srt_0097": "pos",
    "d2.srt_0001": "neu", "d2.srt_0002": "neu", "d2.srt_0005": "neu", "d2.srt_0010": "neg",
    "d2.srt_0014": "neu", "d2.srt_0022": "neu", "d2.srt_0030": "neu", "d2.srt_0036": "neg",
    "d2.srt_0041": "neu", "d2.srt_0045": "neu", "d2.srt_0055": "neg", "d2.srt_0061": "neg",
    "d2.srt_0070": "neu", "d2.srt_0083": "neu", "d2.srt_0087": "neu", "d2.srt_0096": "neg",
    # e1.srt — corruption documentary: harms/gaps/declining score = neg; definitions = neu
    "e1.srt_0001": "neg", "e1.srt_0005": "neg", "e1.srt_0009": "neu", "e1.srt_0013": "neg",
    "e1.srt_0017": "neg", "e1.srt_0021": "neg", "e1.srt_0025": "neu", "e1.srt_0029": "neu",
    "e1.srt_0033": "neg", "e1.srt_0037": "neg", "e1.srt_0041": "neg", "e1.srt_0045": "neg",
    "e1.srt_0049": "neu", "e1.srt_0053": "neu", "e1.srt_0057": "neg", "e1.srt_0061": "neg",
    # f1.srt — Maghreb history: Moroccan victories/empire framed positively = pos (applied
    # CONSISTENTLY, unlike the training 9.srt narration); setbacks/plots = neg; narration = neu
    "f1.srt_0001": "neu", "f1.srt_0007": "neu", "f1.srt_0019": "neu", "f1.srt_0029": "neu",
    "f1.srt_0039": "neu", "f1.srt_0047": "neu", "f1.srt_0055": "neu", "f1.srt_0067": "pos",
    "f1.srt_0079": "neu", "f1.srt_0091": "neu", "f1.srt_0103": "neg", "f1.srt_0111": "pos",
    "f1.srt_0123": "neg", "f1.srt_0135": "pos", "f1.srt_0143": "pos", "f1.srt_0155": "neu",
}


def cohen_kappa(y1, y2, labels=("neg", "neu", "pos")):
    n = len(y1)
    po = sum(a == b for a, b in zip(y1, y2)) / n
    pe = sum((y1.count(l) / n) * (y2.count(l) / n) for l in labels)
    return (po - pe) / (1 - pe) if pe < 1 else 1.0, po


def main():
    df = pd.read_csv(SRC)
    assert set(df["utterance_id"]) == set(V3), "V3 must label exactly the 194 v2 utterances"

    v2 = dict(zip(df["utterance_id"], df["label"]))
    df_v3 = df.copy()
    df_v3["label"] = df_v3["utterance_id"].map(V3)
    df_v3["label_source"] = "claude-rubric-v3"
    df_v3.to_csv(OUT, index=False)

    ids = list(df["utterance_id"])
    y2 = [v2[i] for i in ids]
    y3 = [V3[i] for i in ids]
    kappa, agree = cohen_kappa(y2, y3)

    from collections import Counter
    print(f"Wrote {len(df_v3)} rows -> {OUT}")
    print(f"  v2 (human) dist : {dict(Counter(y2))}")
    print(f"  v3 (claude) dist: {dict(Counter(y3))}")
    print(f"  agreement       : {agree*100:.1f}%   Cohen's kappa: {kappa:.3f}")
    changed = [(i, v2[i], V3[i]) for i in ids if v2[i] != V3[i]]
    print(f"  changed labels  : {len(changed)} / {len(ids)}")
    print(f"  change types    : {dict(Counter((a, b) for _, a, b in changed))}")
    # how the positive class moved
    pos_v2 = {i for i in ids if v2[i] == "pos"}
    pos_v3 = {i for i in ids if V3[i] == "pos"}
    print(f"  pos: v2={len(pos_v2)}  v3={len(pos_v3)}  kept={len(pos_v2 & pos_v3)}  "
          f"v2pos->v3neu/neg={len(pos_v2 - pos_v3)}")


if __name__ == "__main__":
    main()
