"""
Augment broadcast_train_raw.csv with 219 additional hand-annotated utterances
(the unsampled remainder of the SRT pool, excluding the frozen test set).

Run:  python src/apply_annotations_broadcast_aug.py
Output: data/test_sets/broadcast_train_raw.csv  (overwritten, 713 rows total)
"""
import json, os
import pandas as pd

DATA_DIR   = os.path.join(os.path.dirname(__file__), "..", "data", "test_sets")
RAW_CSV    = os.path.join(DATA_DIR, "domaine_reel_raw.csv")
TRAIN_CSV  = os.path.join(DATA_DIR, "broadcast_train_raw.csv")
FROZEN_CSV = os.path.join(DATA_DIR, "domaine_reel_v2.csv")

# ── New annotations for the 219 previously unsampled utterances ──────────────

LABELS: dict[str, str] = {

    # ── 10.srt (7 remaining — health sector) ─────────────────────────────────
    "10.srt_0063": "neu",  # state can't regulate private sector prices — policy framing
    "10.srt_0064": "neg",  # medications eating pockets; bigger problem than HR
    "10.srt_0066": "neg",  # state doesn't invest; private sector fills gap (structural failure)
    "10.srt_0067": "pos",  # dialysis pricing appropriate; medical care accessible
    "10.srt_0068": "neg",  # costs still a burden despite treatment availability
    "10.srt_0069": "neu",  # higher health authority / local agencies — institutional
    "10.srt_0070": "neu",  # meta / call to action

    # ── 2.srt (108 remaining — garbled ASR, pharma/health/social) ─────────────
    "2.srt_0028": "neu",
    "2.srt_0032": "neu",
    "2.srt_0052": "neu",
    "2.srt_0055": "neu",
    "2.srt_0074": "neu",
    "2.srt_0099": "neu",
    "2.srt_0109": "neu",
    "2.srt_0127": "neu",
    "2.srt_0134": "neu",
    "2.srt_0135": "neu",
    "2.srt_0147": "neu",
    "2.srt_0154": "neu",
    "2.srt_0157": "neu",
    "2.srt_0165": "neu",
    "2.srt_0204": "neu",
    "2.srt_0211": "neu",
    "2.srt_0213": "neu",
    "2.srt_0228": "neu",
    "2.srt_0242": "neu",
    "2.srt_0247": "neu",
    "2.srt_0254": "neu",
    "2.srt_0256": "neu",
    "2.srt_0261": "neg",  # elderly with chronic diseases, diabetes, hypertension
    "2.srt_0267": "neg",  # light-sensitivity / medical side-effect problem
    "2.srt_0306": "neu",
    "2.srt_0307": "neu",
    "2.srt_0320": "neu",
    "2.srt_0323": "neu",
    "2.srt_0329": "neu",
    "2.srt_0343": "neu",
    "2.srt_0379": "neu",
    "2.srt_0395": "neu",
    "2.srt_0421": "neu",
    "2.srt_0423": "neu",
    "2.srt_0425": "neu",
    "2.srt_0433": "neu",
    "2.srt_0435": "neu",
    "2.srt_0448": "neu",
    "2.srt_0468": "neu",
    "2.srt_0469": "neu",
    "2.srt_0474": "neu",
    "2.srt_0475": "neu",
    "2.srt_0478": "neu",
    "2.srt_0481": "neu",
    "2.srt_0490": "neu",
    "2.srt_0492": "neu",
    "2.srt_0515": "neu",
    "2.srt_0516": "neu",
    "2.srt_0523": "neu",
    "2.srt_0526": "neg",  # cooking methods destroy Moroccan dietary nutrition
    "2.srt_0536": "neu",
    "2.srt_0537": "neu",
    "2.srt_0544": "neu",
    "2.srt_0546": "neu",
    "2.srt_0550": "neu",
    "2.srt_0551": "neu",
    "2.srt_0557": "neu",
    "2.srt_0558": "neu",
    "2.srt_0566": "neu",
    "2.srt_0567": "neu",
    "2.srt_0570": "neu",
    "2.srt_0579": "neu",
    "2.srt_0583": "neu",
    "2.srt_0590": "neu",
    "2.srt_0597": "neu",
    "2.srt_0598": "neu",
    "2.srt_0602": "neu",
    "2.srt_0606": "neu",
    "2.srt_0613": "neu",
    "2.srt_0615": "neu",
    "2.srt_0623": "neu",
    "2.srt_0629": "neu",
    "2.srt_0634": "neu",
    "2.srt_0660": "neu",
    "2.srt_0663": "neg",  # smoking costs Morocco ~5B DH/year
    "2.srt_0678": "neu",
    "2.srt_0684": "neu",
    "2.srt_0694": "neg",  # heart and artery diseases due to smoking
    "2.srt_0703": "neu",
    "2.srt_0704": "neu",
    "2.srt_0709": "neu",  # World AIDS Day — informational
    "2.srt_0712": "neu",
    "2.srt_0720": "neg",  # artery diseases and cancer linked
    "2.srt_0727": "neu",
    "2.srt_0739": "neu",
    "2.srt_0741": "neu",
    "2.srt_0743": "neu",
    "2.srt_0744": "neu",
    "2.srt_0759": "neg",  # problem: law enforcement / weak implementation
    "2.srt_0772": "neu",
    "2.srt_0790": "neu",
    "2.srt_0795": "pos",  # psychological support programs and social policies implemented
    "2.srt_0799": "neu",
    "2.srt_0813": "neu",
    "2.srt_0814": "neu",
    "2.srt_0816": "neu",
    "2.srt_0835": "neu",
    "2.srt_0841": "neu",
    "2.srt_0863": "neu",
    "2.srt_0875": "neu",
    "2.srt_0903": "neu",
    "2.srt_0904": "neu",
    "2.srt_0931": "neu",
    "2.srt_0951": "neu",
    "2.srt_0979": "neu",
    "2.srt_0992": "neu",
    "2.srt_1001": "neu",
    "2.srt_1002": "neu",

    # ── 5.srt (2 remaining — stock market) ────────────────────────────────────
    "5.srt_0048": "pos",  # long-term investors are likely to be profitable
    "5.srt_0049": "neu",  # advisory disclaimer; consult specialist

    # ── 6.srt (47 remaining — Sufi/religious content, mostly garbled) ─────────
    "6.srt_0018": "pos",  # Atlas Lions (Moroccan football team) won at the World Cup
    "6.srt_0032": "neu",
    "6.srt_0045": "neu",
    "6.srt_0062": "neu",
    "6.srt_0073": "neu",
    "6.srt_0085": "neu",
    "6.srt_0101": "neu",
    "6.srt_0106": "neu",
    "6.srt_0144": "neu",
    "6.srt_0180": "neu",
    "6.srt_0217": "neu",
    "6.srt_0224": "neu",
    "6.srt_0230": "neu",
    "6.srt_0232": "neu",
    "6.srt_0239": "neu",
    "6.srt_0244": "neu",
    "6.srt_0263": "neu",
    "6.srt_0267": "neu",
    "6.srt_0270": "neu",
    "6.srt_0291": "neu",
    "6.srt_0302": "neu",
    "6.srt_0309": "neu",
    "6.srt_0312": "neu",
    "6.srt_0325": "neu",
    "6.srt_0341": "neu",
    "6.srt_0344": "neu",
    "6.srt_0366": "neu",
    "6.srt_0379": "neu",
    "6.srt_0391": "neu",
    "6.srt_0398": "neu",
    "6.srt_0401": "neu",
    "6.srt_0403": "neu",
    "6.srt_0415": "neu",
    "6.srt_0425": "neu",
    "6.srt_0435": "neu",
    "6.srt_0439": "neu",
    "6.srt_0445": "neu",
    "6.srt_0448": "neu",
    "6.srt_0454": "neu",
    "6.srt_0467": "neu",
    "6.srt_0473": "neu",
    "6.srt_0531": "pos",  # congratulations to all winners
    "6.srt_0543": "pos",  # excellent tourist city, beautiful (praise)
    "6.srt_0589": "neu",
    "6.srt_0637": "neu",
    "6.srt_0639": "neu",
    "6.srt_0657": "neu",

    # ── 7.srt (8 remaining — Western Sahara / Morocco history) ────────────────
    "7.srt_0047": "pos",  # UN Security Council Res. 2797: no rejections; 3 permanent members voted yes
    "7.srt_0048": "neg",  # beginning of bigger challenge: Tindouf camps, tens of thousands displaced
    "7.srt_0049": "pos",  # real political participation for Sahrawis, including former Polisario members
    "7.srt_0050": "neg",  # 50 years after Green March: victory but underlying problems remain
    "7.srt_0051": "neu",  # book on Morocco's eastern borders — publication announcement
    "7.srt_0052": "neu",  # crowdfunding / production support appeal
    "7.srt_0053": "pos",  # Morocco's future scenarios: will further consolidate its position
    "7.srt_0054": "neu",  # thank-you / outro to historian

    # ── 7769.srt (20 remaining — PJD party internal & political debate) ───────
    "7769.srt_0052": "pos",  # PJD achieved the Casablanca desalination deal despite being in opposition
    "7769.srt_0053": "neu",  # law regulates these operations — legal framing
    "7769.srt_0054": "pos",  # investment committee will receive and support the desalination file
    "7769.srt_0055": "neg",  # impossible to be on both sides at once — conflict of interest charge
    "7769.srt_0056": "neg",  # big political problem: opposition vs. government; potential legal violations
    "7769.srt_0057": "neu",  # 9th national congress background
    "7769.srt_0058": "neg",  # absence of many founding leaders who left PJD since inception
    "7769.srt_0059": "neu",  # framing around the question — procedural
    "7769.srt_0060": "neu",  # party as institution made no disciplinary decisions
    "7769.srt_0061": "pos",  # party built on freedom; door open for former members to return
    "7769.srt_0062": "neu",  # each member decided individually to resign — neutral explanation
    "7769.srt_0063": "neu",  # PJD has national and partisan responsibilities
    "7769.srt_0064": "neu",  # key challenges: citizen organization, oversight
    "7769.srt_0065": "neg",  # pre-congress: party members delivered self-critical speeches
    "7769.srt_0066": "neg",  # crisis speech: shock from 2021 election results
    "7769.srt_0067": "pos",  # entering elections with great hope; genuine work done before congress
    "7769.srt_0069": "neg",  # pre-election coordination extremely difficult to arrange
    "7769.srt_0070": "neg",  # never coordinated with USFP; political tensions exposed
    "7769.srt_0071": "neg",  # "World Cup government" label mocked — government credibility problem
    "7769.srt_0072": "neu",  # outro / thank-you

    # ── 7770.srt (27 remaining — USFP party & government critique) ────────────
    "7770.srt_0055": "neu",  # 7 months until congress; party organizing
    "7770.srt_0056": "neu",  # will answer fundamental question in September
    "7770.srt_0058": "pos",  # party is in good health and on ascending trajectory for 2026
    "7770.srt_0059": "pos",  # elected to Socialist International and youth international secretariat
    "7770.srt_0060": "pos",  # achieved advanced results for Morocco, especially on Sahara
    "7770.srt_0062": "neu",  # congress preparation described as normal and organized
    "7770.srt_0063": "neu",  # USFP will choose the best candidate for current phase
    "7770.srt_0064": "neu",  # question about relationship between opposition and majority party
    "7770.srt_0065": "neg",  # why no good relationship with PJD?
    "7770.srt_0066": "neg",  # PJD damaged the relationship; history of bad governance
    "7770.srt_0067": "neg",  # PJD fears USFP taking their middle-class constituency
    "7770.srt_0071": "pos",  # party is ready and prepared to enter government
    "7770.srt_0072": "neu",  # 3-year record presented; people are the judge
    "7770.srt_0073": "neg",  # growth: promised 4%, hasn't exceeded 2%
    "7770.srt_0074": "neg",  # can't achieve targets; 1M jobs not created; unemployment spread
    "7770.srt_0076": "neg",  # middle class destroyed; only rich remain
    "7770.srt_0078": "neg",  # 12M Moroccans get only 500 DH social protection — insufficient
    "7770.srt_0079": "neg",  # youth crisis: construction sites or prison — no real path
    "7770.srt_0080": "neg",  # legislative proposals rejected; education promises broken
    "7770.srt_0081": "neg",  # price increases justified with excuses: drought, Ukraine, COVID
    "7770.srt_0082": "neg",  # one person supports 30 through investment — system inefficiency
    "7770.srt_0084": "neg",  # subsidies go to rich (companies) not poor
    "7770.srt_0085": "pos",  # most active parliamentary group; always present; tracked absenteeism
    "7770.srt_0086": "neg",  # forced to name absent ministers publicly in parliament
    "7770.srt_0087": "neg",  # race to "World Cup government" started early — premature electioneering
    "7770.srt_0088": "neg",  # election integrity problem raised with Interior Minister
    "7770.srt_0089": "pos",  # guest thanks host; congratulates show on returning live
}


def main() -> None:
    raw    = pd.read_csv(RAW_CSV)
    frozen = pd.read_csv(FROZEN_CSV)
    existing = pd.read_csv(TRAIN_CSV)

    used = set(existing["utterance_id"]) | set(frozen["utterance_id"])
    pool = raw[~raw["utterance_id"].isin(used)].copy()

    missing = [uid for uid in pool["utterance_id"] if uid not in LABELS]
    extra   = [uid for uid in LABELS if uid not in set(pool["utterance_id"])]
    if missing:
        print(f"WARNING: {len(missing)} pool IDs not in LABELS: {missing[:5]}")
    if extra:
        print(f"WARNING: {len(extra)} LABELS keys not in pool: {extra[:5]}")

    new_rows = []
    for _, row in pool.iterrows():
        label = LABELS.get(row["utterance_id"])
        if label is None:
            continue
        new_rows.append({
            "utterance_id": row["utterance_id"],
            "file":         row["utterance_id"].split("_")[0],
            "fmt":          "srt",
            "detected_lang": "ar",
            "text":          row["text"],
            "label":         label,
        })

    new_df   = pd.DataFrame(new_rows)
    combined = pd.concat([existing, new_df], ignore_index=True)

    lc = combined["label"].value_counts()
    lc_new = new_df["label"].value_counts()
    print(f"New rows  : {len(new_df)}  "
          f"(neg={lc_new.get('neg',0)}, pos={lc_new.get('pos',0)}, neu={lc_new.get('neu',0)})")
    print(f"Combined  : {len(combined)}  "
          f"(neg={lc.get('neg',0)}, pos={lc.get('pos',0)}, neu={lc.get('neu',0)})")

    combined.to_csv(TRAIN_CSV, index=False)
    print(f"Written → {TRAIN_CSV}")


if __name__ == "__main__":
    main()
