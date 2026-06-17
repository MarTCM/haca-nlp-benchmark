"""
Apply manual sentiment annotations to the extracted utterances.

Reads data/test_sets/domaine_reel_raw.csv, fills the `label` column from the
LABELS dict below (keyed by utterance_id), and writes the frozen test set to
data/test_sets/domaine_reel.csv in the standard [text, label] schema (plus
provenance columns).

Labels: neg / neu / pos
Annotation rules are documented in report/DOMAINE_REEL_ANNOTATION.md
"""

import csv
import hashlib
import os

RAW = "data/test_sets/domaine_reel_raw.csv"
OUT = "data/test_sets/domaine_reel.csv"

LABELS = {
    # ---- 1.srt : ASR-garbled news fragments — sentiment indeterminate → neu ----
    "1.srt_0007": "neu",
    "1.srt_0095": "neu",
    "1.srt_0135": "neu",
    "1.srt_0216": "neu",
    "1.srt_0296": "pos",   # "خليهم يفرحوا" — let them rejoice
    "1.srt_0306": "neu",
    "1.srt_0327": "neg",   # "مسكين" — pity / poor man
    "1.srt_0441": "neu",
    "1.srt_0508": "neu",
    "1.srt_0538": "neu",
    "1.srt_0560": "neu",

    # ---- 10.srt : health sector — informational + heavy critique of failures ----
    "10.srt_0001": "neg",  # 63% out-of-pocket health spending = high burden
    "10.srt_0005": "neu",  # intro + definition of health
    "10.srt_0009": "neu",  # definition of health system
    "10.srt_0013": "neu",  # describes mobile units / free access
    "10.srt_0017": "neu",  # lists university hospitals
    "10.srt_0021": "neu",  # national tariff agency
    "10.srt_0025": "neu",  # emergency centres
    "10.srt_0029": "neu",  # rehabilitation centres
    "10.srt_0033": "neu",  # clinics / profit motive
    "10.srt_0037": "neg",  # "both sectors have big problems → people protested"
    "10.srt_0041": "neg",  # need to double the workforce; big shortage
    "10.srt_0045": "neg",  # doctor emigration, "they hate it"
    "10.srt_0049": "neg",  # budget shortfall, losing 6%
    "10.srt_0053": "neg",  # centralised, complex, slow
    "10.srt_0057": "neg",  # ambulances don't come, legal confusion
    "10.srt_0061": "neg",  # clinic holds patient hostage for guarantee cheque
    "10.srt_0065": "neu",  # private sector as a proposed solution

    # ---- 2.srt : ASR-garbled medical/pharma fragments → mostly neu ----
    "2.srt_0017": "neu",
    "2.srt_0062": "neu",
    "2.srt_0138": "neu",
    "2.srt_0212": "neu",
    "2.srt_0257": "neu",   # "أهلا بك ومرحبا" — greeting
    "2.srt_0328": "neu",
    "2.srt_0424": "neu",
    "2.srt_0473": "neg",   # "للأسف مبقتش..." — unfortunately no longer
    "2.srt_0497": "neu",
    "2.srt_0538": "neu",
    "2.srt_0559": "neu",
    "2.srt_0593": "neu",
    "2.srt_0619": "neu",
    "2.srt_0681": "neu",
    "2.srt_0717": "neu",   # heart diseases — informational
    "2.srt_0747": "neu",   # smoking ban on trains — informational
    "2.srt_0816": "neu",   # applying the law — informational

    # ---- 3.srt : 2025 tax law — explainer with critique of inequality ----
    "3.srt_0001": "neu",
    "3.srt_0003": "neu",
    "3.srt_0005": "neu",
    "3.srt_0007": "neu",
    "3.srt_0009": "neg",   # middle class excluded; reforms favour the rich
    "3.srt_0011": "neu",
    "3.srt_0013": "neu",
    "3.srt_0015": "neu",
    "3.srt_0017": "neu",
    "3.srt_0019": "neu",
    "3.srt_0021": "neg",   # tax injustice "تناقض" contradiction
    "3.srt_0023": "neu",
    "3.srt_0025": "neu",
    "3.srt_0027": "neu",
    "3.srt_0029": "neg",   # workers didn't benefit in reality
    "3.srt_0031": "neu",
    "3.srt_0033": "neu",

    # ---- 4.srt : public procurement — practical explainer, mostly neutral ----
    "4.srt_0001": "neu",
    "4.srt_0004": "neu",
    "4.srt_0007": "neg",   # baksheesh / corruption keeps people out
    "4.srt_0010": "neu",
    "4.srt_0013": "neu",
    "4.srt_0016": "neu",
    "4.srt_0019": "neu",
    "4.srt_0022": "neu",
    "4.srt_0025": "neu",
    "4.srt_0028": "neu",
    "4.srt_0031": "neu",
    "4.srt_0034": "neu",
    "4.srt_0037": "neu",
    "4.srt_0040": "neu",
    "4.srt_0043": "neu",
    "4.srt_0046": "neu",
    "4.srt_0049": "neu",

    # ---- 5.srt : stock market — explainer + success/failure anecdotes ----
    "5.srt_0001": "pos",   # shares multiplied 7x — gains
    "5.srt_0003": "neg",   # bank manager belittled small investor "حقاره"
    "5.srt_0005": "neu",
    "5.srt_0007": "neu",
    "5.srt_0009": "neu",
    "5.srt_0011": "neu",
    "5.srt_0013": "neu",
    "5.srt_0015": "neu",
    "5.srt_0017": "neu",   # crypto volatility — informational
    "5.srt_0019": "neu",
    "5.srt_0021": "neu",   # Casablanca exchange small vs world
    "5.srt_0023": "neg",   # companies avoid transparency
    "5.srt_0025": "neu",
    "5.srt_0027": "neu",
    "5.srt_0029": "neu",
    "5.srt_0031": "neu",
    "5.srt_0033": "neu",

    # ---- 6.srt : ASR-garbled religious/Sufi + misc → neu ----
    "6.srt_0001": "neu",
    "6.srt_0045": "neu",
    "6.srt_0079": "neu",
    "6.srt_0106": "neu",
    "6.srt_0209": "neu",
    "6.srt_0230": "neu",
    "6.srt_0243": "neu",   # Sufism / ethics
    "6.srt_0267": "neu",   # basmala invocation
    "6.srt_0297": "neu",
    "6.srt_0312": "neu",   # "weigh with justice" — Quranic
    "6.srt_0342": "neu",
    "6.srt_0379": "neu",
    "6.srt_0399": "neu",
    "6.srt_0415": "neu",
    "6.srt_0436": "neu",
    "6.srt_0448": "neu",
    "6.srt_0469": "neu",

    # ---- 7.srt : Sahara history — patriotic, celebratory ----
    "7.srt_0001": "pos",   # unity day; ends "141 years of humiliation"
    "7.srt_0004": "neu",   # investment advice / sponsorship
    "7.srt_0007": "neu",   # historical decline narrative
    "7.srt_0010": "neu",   # bay'a allegiance system
    "7.srt_0013": "neu",   # Portuguese occupation history
    "7.srt_0016": "neu",   # French taking territory
    "7.srt_0019": "neu",   # colonisation history
    "7.srt_0022": "neu",   # Tarfaya recovery
    "7.srt_0025": "neu",   # Eastern Sahara dispute
    "7.srt_0028": "neu",   # self-determination / ICJ opinion
    "7.srt_0031": "neu",   # Polisario movements
    "7.srt_0034": "pos",   # "decisively resolved for Morocco"
    "7.srt_0037": "pos",   # Morocco building roads/universities — development
    "7.srt_0040": "pos",   # UN praised initiative as "serious and realistic"
    "7.srt_0043": "pos",   # US recognition strengthens Morocco
    "7.srt_0046": "pos",   # French recognition; European support
    "7.srt_0049": "pos",   # national reconciliation, dignity

    # ---- 7769.srt : political debate (PJD opposition) — critique ----
    "7769.srt_0001": "neu",  # show intro
    "7769.srt_0006": "neu",  # greeting guests
    "7769.srt_0010": "neu",  # explains censure motion
    "7769.srt_0014": "neu",  # internal party debate
    "7769.srt_0018": "neg",  # "you failed in the fact-finding committee"
    "7769.srt_0022": "neg",  # govt incapable; 4% growth criticism
    "7769.srt_0027": "neu",  # agricultural middle-class stats
    "7769.srt_0031": "neg",  # ideological critique of tax policy
    "7769.srt_0037": "neg",  # importers benefited from exemptions
    "7769.srt_0042": "neg",  # subsidy distribution / partisanship
    "7769.srt_0046": "neg",  # heated exchange "ابنادم ازرق ما فاهم"
    "7769.srt_0051": "neg",  # no political will to reform pensions
    "7769.srt_0055": "neu",  # "let's be clear"
    "7769.srt_0059": "neu",  # answering, "no agenda"
    "7769.srt_0063": "neu",  # party responsibility / experience
    "7769.srt_0067": "pos",  # electoral program, "great hope"

    # ---- 7770.srt : political debate — critique ----
    "7770.srt_0001": "neu",  # show intro
    "7770.srt_0006": "neu",  # "thanks for hosting"
    "7770.srt_0010": "neu",  # party rejection / coordination narrative
    "7770.srt_0015": "neu",  # fact-finding committee, parties agreeing
    "7770.srt_0019": "neu",  # meeting logistics
    "7770.srt_0023": "neu",  # internal disagreement narrative
    "7770.srt_0029": "neu",  # "we were waiting, nothing happened"
    "7770.srt_0033": "neg",  # accusations: deal/embezzlement/kidnapping
    "7770.srt_0039": "neu",  # heated "let him bury it"
    "7770.srt_0044": "neu",  # French parliament censure comparison
    "7770.srt_0049": "neg",  # criticism of ex-PM's "empty talk"
    "7770.srt_0053": "neg",  # "we failed... we regret"
    "7770.srt_0059": "neu",  # party elected in Socialist International
    "7770.srt_0064": "neu",  # inter-party relations
    "7770.srt_0071": "neu",  # party ready to govern
    "7770.srt_0076": "neg",  # "middle class killed; what did you do for the poor?"

    # ---- 8.srt : corruption — critical throughout ----
    "8.srt_0001": "neg",   # 115bn embezzled from social-security fund
    "8.srt_0003": "neg",   # corruption ranking dropped
    "8.srt_0005": "neu",   # sponsorship appeal
    "8.srt_0007": "neg",   # nepotism / "corruption you did"
    "8.srt_0009": "neg",   # Morocco low on corruption index, "backward"
    "8.srt_0011": "neg",   # "loyalty becomes a value" — clientelism
    "8.srt_0013": "neu",   # investment-efficiency index explainer
    "8.srt_0015": "neu",   # legal definition of corruption
    "8.srt_0017": "neg",   # anti-corruption strategy failed
    "8.srt_0019": "neg",   # legislative gap on corruption
    "8.srt_0021": "neg",   # "dangerous ambiguity" in asset-declaration decree
    "8.srt_0023": "neg",   # too few auditors; "big legislative deficiency"
    "8.srt_0025": "neu",   # presumption-of-innocence principle
    "8.srt_0027": "neu",   # illicit-enrichment convention
    "8.srt_0029": "neg",   # conflict of interest = biggest scandal source
    "8.srt_0031": "neg",   # missing conflict-of-interest law

    # ---- 9.srt : Maghreb history — narrative, neutral ----
    "9.srt_0001": "neu",
    "9.srt_0004": "neu",
    "9.srt_0007": "neu",
    "9.srt_0010": "neu",
    "9.srt_0013": "neu",
    "9.srt_0016": "neu",
    "9.srt_0019": "neu",
    "9.srt_0022": "neu",
    "9.srt_0025": "neu",
    "9.srt_0028": "neu",
    "9.srt_0031": "neu",
    "9.srt_0034": "neu",
    "9.srt_0037": "neu",
    "9.srt_0040": "neu",
    "9.srt_0043": "neu",
    "9.srt_0046": "neu",
}


def main():
    with open(RAW, encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))

    missing = [r["utterance_id"] for r in rows if r["utterance_id"] not in LABELS]
    if missing:
        raise SystemExit(f"Missing labels for {len(missing)} ids: {missing[:10]} ...")

    for r in rows:
        r["label"] = LABELS[r["utterance_id"]]

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    fieldnames = ["utterance_id", "file", "fmt", "detected_lang", "text", "label"]
    with open(OUT, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow({k: r[k] for k in fieldnames})

    # Distribution + hash
    from collections import Counter
    dist = Counter(r["label"] for r in rows)
    with open(OUT, "rb") as fh:
        h = hashlib.sha256(fh.read()).hexdigest()

    print(f"Wrote {len(rows)} labelled utterances → {OUT}")
    print(f"  neg={dist['neg']}  neu={dist['neu']}  pos={dist['pos']}")
    print(f"  sha256={h}")


if __name__ == "__main__":
    main()
