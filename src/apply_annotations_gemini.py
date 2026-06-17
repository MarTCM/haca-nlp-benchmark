"""
Apply re-annotations to gemini.csv utterances using the broader philosophy:
"Does this utterance carry negative/positive CONTENT, regardless of presenter tone?"

neg = describes something bad: failure, burden, corruption, inequality, conflict, shortage
pos = describes something good: success, achievement, progress, opportunity, optimism
neu = purely procedural/definitional/structural — no evaluative dimension

Writes data/test_sets/domaine_reel_v2.csv
"""

import csv
import hashlib
import os
from collections import Counter

IN  = "data/test_sets/gemini.csv"
OUT = "data/test_sets/domaine_reel_v2.csv"

LABELS = {
    # ── 1.srt — ASR-garbled: no recoverable signal → neu; clear words decide ──
    "1.srt_0007": "neu",   # garbled
    "1.srt_0095": "neu",   # garbled
    "1.srt_0135": "neu",   # garbled
    "1.srt_0216": "neu",   # garbled
    "1.srt_0296": "pos",   # "خليهم يفرحوا" — let them rejoice
    "1.srt_0306": "neu",   # garbled
    "1.srt_0327": "neg",   # مسكين + money hardship
    "1.srt_0441": "neu",   # hypothetical setup
    "1.srt_0508": "neg",   # suspicion, disrupting the session
    "1.srt_0538": "neg",   # العصابة (gang)
    "1.srt_0560": "neu",   # garbled

    # ── 10.srt — health sector ───────────────────────────────────────────────
    "10.srt_0001": "neg",  # 63% out-of-pocket: heavy financial burden on citizens
    "10.srt_0005": "neg",  # explicitly says "الروينه ديالو كامله" (complete collapse)
    "10.srt_0009": "neu",  # health system legal definition
    "10.srt_0013": "pos",  # mobile health units providing free care in rural areas
    "10.srt_0017": "neu",  # university hospitals listing
    "10.srt_0021": "neu",  # national tariff agency
    "10.srt_0025": "neu",  # emergency centers
    "10.srt_0029": "neu",  # addiction/rehabilitation centers listing
    "10.srt_0033": "neu",  # private clinics for-profit vs non-profit
    "10.srt_0037": "neg",  # both sectors have big problems; people protested
    "10.srt_0041": "neg",  # need 100k more workers; government failed to solve it
    "10.srt_0045": "neg",  # doctor emigration — half of Moroccan doctors work abroad
    "10.srt_0049": "neg",  # health budget at 6%, needs 12%; chronic shortfall
    "10.srt_0053": "neg",  # over-centralized, hospitals can't act independently
    "10.srt_0057": "neg",  # ambulances don't reach homes; legal deadlock
    "10.srt_0061": "neg",  # clinic holds newborn hostage until guarantee cheque paid
    "10.srt_0065": "pos",  # new law opens health investment to non-doctors → more supply

    # ── 11.srt — garbled ASR ─────────────────────────────────────────────────
    "11.srt_0059": "neu",  # garbled
    "11.srt_0182": "neg",  # references killing / tainted relations
    "11.srt_0190": "neu",  # upbringing/education
    "11.srt_0209": "neu",  # garbled

    # ── 12.srt — medical/pharma (garbled) ────────────────────────────────────
    "12.srt_0016": "neu",  # greeting
    "12.srt_0097": "neu",  # pharmaceutical distribution, unclear
    "12.srt_0109": "neg",  # medical errors inside Morocco
    "12.srt_0171": "neg",  # "للأسف مبقتش تلك الكميات الفيتامينات" — vitamin shortage

    # ── 13.srt — garbled ASR ─────────────────────────────────────────────────
    "13.srt_0069": "neu",
    "13.srt_0101": "neu",
    "13.srt_0110": "neu",
    "13.srt_0192": "neu",  # garbled
    "13.srt_0219": "neg",  # references conflict / جرحى (wounded)
    "13.srt_0284": "neu",
    "13.srt_0313": "neu",  # heart disease — neutral medical mention

    # ── 14.srt — smoking ban on trains ───────────────────────────────────────
    "14.srt_0006": "pos",  # national rail smoking ban — positive public health measure
    "14.srt_0050": "neu",  # applying the law — procedural

    # ── 2.srt — 2025 tax law explainer ───────────────────────────────────────
    "2.srt_0001": "neg",   # new tax rules described as a headache for content creators
    "2.srt_0005": "neu",   # intro / sponsorship
    "2.srt_0009": "neg",   # 2019 conference concluded tax system is "مرون" (unfair)
    "2.srt_0013": "neu",   # tax types and amounts — procedural
    "2.srt_0017": "neg",   # tax reform benefited everyone except the middle class
    "2.srt_0021": "neu",   # tax residency rules
    "2.srt_0025": "neu",   # tax brackets (0–38%) — procedural
    "2.srt_0029": "neu",   # flat-rate system for self-employed
    "2.srt_0033": "neu",   # withholding tax on property income
    "2.srt_0037": "neu",   # double taxation on dividends — procedural explanation
    "2.srt_0041": "neg",   # same income, different tax rates = tax inequality
    "2.srt_0045": "neu",   # government counter-argument on effective rates
    "2.srt_0049": "neu",   # effective rate 17% not 34% — procedural clarification
    "2.srt_0053": "neg",   # critics say reform did nothing for workers
    "2.srt_0057": "neg",   # workers in reality didn't benefit
    "2.srt_0061": "neu",   # withholding enforcement — procedural
    "2.srt_0065": "neu",   # tax audit procedure

    # ── 3.srt — public procurement guide ─────────────────────────────────────
    "3.srt_0001": "pos",   # government contracts framed as 245bn opportunity for citizens
    "3.srt_0047": "neu",   # participatory finance product
    "3.srt_0055": "neg",   # insider connections used to win contracts (corruption)
    "3.srt_0059": "neu",   # types of government needs: works, supplies, services
    "3.srt_0085": "neu",   # payment schedule options
    "3.srt_0089": "neu",   # contract types (tranches, framework)
    "3.srt_0093": "neu",   # shared contracts
    "3.srt_0109": "neu",   # IP protection advice — procedural
    "3.srt_0113": "neu",   # procurement methods
    "3.srt_0125": "pos",   # simplified dossier introduced in 2023 for newcomers
    "3.srt_0129": "neu",   # how to find tenders on government portal
    "3.srt_0133": "neu",   # tender announcement structure
    "3.srt_0153": "neu",   # disqualification for errors — procedural warning
    "3.srt_0157": "neu",   # complaint procedure
    "3.srt_0169": "neu",   # price preference for national products
    "3.srt_0177": "neu",   # supplier payment portal
    "3.srt_0185": "neu",   # negotiated contracts

    # ── 4.srt — stock market explainer ───────────────────────────────────────
    "4.srt_0001": "pos",   # shares multiplied 7× — investment success story
    "4.srt_0005": "neg",   # bank manager humiliated small investor ("حسستني بحقاره")
    "4.srt_0009": "neu",   # Dutch East India Company historical example
    "4.srt_0013": "neu",   # how shares trade between investors
    "4.srt_0025": "neu",   # bonds explanation
    "4.srt_0045": "neu",   # commodities markets
    "4.srt_0049": "neu",   # futures contracts for farmers
    "4.srt_0057": "neu",   # speculation — neutral educational explanation
    "4.srt_0065": "neg",   # crypto described as high-risk ("فيه خطر كبير")
    "4.srt_0069": "neu",   # indices and mutual funds
    "4.srt_0081": "neg",   # Casablanca stock exchange tiny vs world ("نقطه في البحر")
    "4.srt_0085": "neg",   # companies avoid listing to avoid transparency
    "4.srt_0101": "neu",   # company valuation method
    "4.srt_0105": "neu",   # listing requirements
    "4.srt_0109": "neu",   # analyst valuation factors
    "4.srt_0113": "neu",   # IPO process
    "4.srt_0121": "neu",   # IPO subscription period

    # ── 5.srt — garbled ASR ──────────────────────────────────────────────────
    "5.srt_0028": "neu",
    "5.srt_0099": "neg",   # غارة (military raid/attack)
    "5.srt_0137": "neg",   # غرامة (fine/penalty)
    "5.srt_0201": "neg",   # "خارج القانون" (outside the law)

    # ── 6.srt — Sufi/religious ────────────────────────────────────────────────
    "6.srt_0110": "neu",
    "6.srt_0116": "neu",
    "6.srt_0121": "neu",

    # ── 7.srt — religious/Sufi ───────────────────────────────────────────────
    "7.srt_0001": "neu",   # Basmala
    "7.srt_0070": "neu",   # garbled
    "7.srt_0150": "neu",   # Quranic injunction — neutral religious
    "7.srt_0153": "neu",   # religious/moral content
    "7.srt_0185": "neu",   # partial/unclear context
    "7.srt_0209": "neu",   # religious values

    # ── 8.srt — garbled ASR ──────────────────────────────────────────────────
    "8.srt_0008": "neu",
    "8.srt_0083": "neu",
    "8.srt_0095": "neg",   # warriors overpowering the weak
    "8.srt_0110": "neu",

    # ── 9.srt — Sahara history ───────────────────────────────────────────────
    "9.srt_0001": "pos",   # national holiday announced; ends 141 years of humiliation
    "9.srt_0005": "neu",   # sponsorship / transition
    "9.srt_0021": "neg",   # internal divisions + foreign threats, state collapsed
    "9.srt_0025": "neu",   # Bay'a system explained
    "9.srt_0033": "neg",   # Portuguese colonial occupation from 15th century
    "9.srt_0037": "neg",   # Moroccan Saharan territories lost to France
    "9.srt_0041": "neg",   # Spanish/French "pacification" operations against locals
    "9.srt_0053": "pos",   # Morocco recovered Tarfaya through direct negotiations
    "9.srt_0065": "neg",   # failed deal; Moroccan people refused to cede Sahara
    "9.srt_0069": "neu",   # ICJ opinion ambiguous — neutral legal analysis
    "9.srt_0081": "neg",   # Polisario military activities, Mauritania destabilized
    "9.srt_0093": "pos",   # wall decisively settled the conflict for Morocco
    "9.srt_0109": "pos",   # Morocco investing in infrastructure and social programs
    "9.srt_0113": "pos",   # UN Security Council: Morocco's proposal "serious and realistic"
    "9.srt_0125": "pos",   # US recognition; military operation success at Guerguarat
    "9.srt_0137": "pos",   # France recognized sovereignty; European consulates opening
    "9.srt_0141": "pos",   # autonomy plan: real political participation + national reconciliation

    # ── d1.srt — political debate episode 1 ──────────────────────────────────
    "d1.srt_0001": "neu",  # show intro
    "d1.srt_0002": "neu",  # guest greeting
    "d1.srt_0007": "neu",  # procedural explanation of censure motion
    "d1.srt_0020": "neu",  # internal party deliberation
    "d1.srt_0028": "neg",  # accuses RNI of corruption: "في كرشهم العجينة"
    "d1.srt_0036": "neg",  # government incapable; only 4% growth over 10 years
    "d1.srt_0045": "neu",  # agricultural middle-class statistics
    "d1.srt_0055": "neg",  # tax reform ideological: hits small, protects big
    "d1.srt_0060": "neg",  # importers pocketed the 13bn DH price-relief subsidies
    "d1.srt_0064": "neg",  # subsidy system: partisan, electoral manipulation, 13bn misallocated
    "d1.srt_0067": "neg",  # heated exchange; insults ("ابنادم ازرق ما فاهم")
    "d1.srt_0078": "neg",  # pension reform: no political will, committees without results
    "d1.srt_0083": "neu",  # debate transition
    "d1.srt_0087": "neu",  # procedural clarification
    "d1.srt_0091": "neu",  # party describing 30-year experience
    "d1.srt_0097": "pos",  # entering elections with great hope ("أمل كبير")

    # ── d2.srt — political debate episode 2 ──────────────────────────────────
    "d2.srt_0001": "neu",  # show intro
    "d2.srt_0002": "neu",  # thanks/welcome
    "d2.srt_0005": "neu",  # factual account of which parties rejected the motion
    "d2.srt_0010": "neg",  # PJD blocking; political deadlock
    "d2.srt_0014": "neu",  # meeting outcome — procedural
    "d2.srt_0022": "neu",  # political narrative — procedural
    "d2.srt_0030": "neg",  # broken commitment; they didn't show up as agreed
    "d2.srt_0036": "neg",  # accusations: صفقة (deal), اختلاس (embezzlement), kidnapping
    "d2.srt_0041": "neu",  # HP party's official statement — procedural
    "d2.srt_0045": "neu",  # comparison with French censure motion — neutral analysis
    "d2.srt_0055": "neg",  # ex-PM's discourse has no place in political thinking
    "d2.srt_0061": "neg",  # failed to create public debate; expresses regret
    "d2.srt_0070": "neu",  # party's international role dismissed
    "d2.srt_0083": "neu",  # Morocco's fluid political coalitions — analysis
    "d2.srt_0087": "neu",  # party ready to enter government
    "d2.srt_0096": "neg",  # "الطبقة المتوسطة تقتلات" + "اش درتي للفقراء؟"

    # ── e1.srt — corruption explainer ────────────────────────────────────────
    "e1.srt_0001": "neg",  # 115bn DH embezzled from social security fund
    "e1.srt_0005": "neg",  # CPI score dropped; minister says corruption unfightable
    "e1.srt_0009": "neu",  # YouTube crowdfunding CTA
    "e1.srt_0013": "neg",  # nepotism/clientelism in hiring described
    "e1.srt_0017": "neg",  # Morocco 37/100 on CPI; linked to underdevelopment
    "e1.srt_0021": "neg",  # loyalty replaces competence; patronage culture
    "e1.srt_0025": "neu",  # ICOR investment-efficiency index — procedural
    "e1.srt_0029": "neu",  # legal definition of corruption — procedural
    "e1.srt_0033": "neg",  # 2015 anti-corruption strategy failed; score dropped 6 points
    "e1.srt_0037": "neg",  # legislative gap in corruption — incomplete legal framework
    "e1.srt_0041": "neg",  # dangerous ambiguity in asset-declaration decree
    "e1.srt_0045": "neg",  # 15 auditors for thousands of declarations; huge gap vs France
    "e1.srt_0049": "neu",  # presumption of innocence principle — legal analysis
    "e1.srt_0053": "neu",  # international convention flexibility — procedural
    "e1.srt_0057": "neg",  # conflict of interest = biggest scandal source; no law
    "e1.srt_0061": "neg",  # conflict of interest law covers regions but not government

    # ── f1.srt — Morocco/Maghreb history ─────────────────────────────────────
    "f1.srt_0001": "neu",  # series intro
    "f1.srt_0007": "neu",  # early North Africa — neutral historical narration
    "f1.srt_0019": "neu",  # Punic Wars, Numidia — neutral historical
    "f1.srt_0029": "neu",  # Roman retreat — neutral historical
    "f1.srt_0039": "neu",  # Tariq ibn Ziyad historical analysis
    "f1.srt_0047": "neu",  # traditional state administration structure
    "f1.srt_0055": "neu",  # local dynasties, Berghouata
    "f1.srt_0067": "pos",  # Almoravid empire: first 100% Moroccan empire
    "f1.srt_0079": "neu",  # Marinid-Zayyanid rivalry — neutral historical
    "f1.srt_0091": "neu",  # Ottoman entry into Algeria — neutral historical
    "f1.srt_0103": "neg",  # Ottoman-Morocco conflict; territorial losses
    "f1.srt_0111": "pos",  # Moroccans united to defeat Ottomans
    "f1.srt_0123": "neg",  # Battle of Three Kings: alarming conditions, poisoning attempt
    "f1.srt_0135": "pos",  # Ahmad al-Mansur's victory; builds Morocco's system
    "f1.srt_0143": "pos",  # Morocco conquers Songhai, secures gold mines
    "f1.srt_0155": "neu",  # Alawi dynasty — mixed historical narration
}


def main():
    with open(IN, encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))

    missing = [r["utterance_id"] for r in rows if r["utterance_id"] not in LABELS]
    if missing:
        raise SystemExit(f"Missing labels for {len(missing)} ids: {missing}")

    for r in rows:
        r["label"] = LABELS[r["utterance_id"]]

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    fieldnames = ["utterance_id", "file", "fmt", "detected_lang", "text", "label"]
    with open(OUT, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow({k: r[k] for k in fieldnames})

    dist = Counter(r["label"] for r in rows)
    import hashlib
    with open(OUT, "rb") as fh:
        h = hashlib.sha256(fh.read()).hexdigest()

    print(f"Wrote {len(rows)} utterances → {OUT}")
    print(f"  neg={dist['neg']}  neu={dist['neu']}  pos={dist['pos']}")
    print(f"  sha256={h}")


if __name__ == "__main__":
    main()
