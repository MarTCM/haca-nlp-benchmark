"""
Apply hand-written annotations (Claude, HACA content-valence philosophy) to the
494 sampled broadcast utterances and write broadcast_train_raw.csv.

Labels: neg / neu / pos
Philosophy:
  neg = describes something bad (failure, burden, corruption, conflict, shortage, injustice)
  pos = describes something good (success, achievement, progress, opportunity, recognition)
  neu = procedural/definitional/structural, garbled ASR, religious recitation, meta
"""
import json
import os
import pandas as pd

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
JSONL_PATH = "/tmp/to_annotate.jsonl"
OUT_PATH = os.path.join(DATA_DIR, "test_sets", "broadcast_train_raw.csv")

# ── Hand annotations (Claude, 2026-06-17) ───────────────────────────────────
# Format: utterance_id → label
LABELS: dict[str, str] = {
    # ── 10.srt  (health sector reporting, 46 utts) ───────────────────────────
    "10.srt_0002": "neg",  # funding gap/patients turned away
    "10.srt_0003": "neg",  # healthcare system failures
    "10.srt_0004": "neu",
    "10.srt_0006": "neu",
    "10.srt_0007": "neu",
    "10.srt_0008": "neu",
    "10.srt_0010": "neu",
    "10.srt_0011": "neu",
    "10.srt_0012": "neu",
    "10.srt_0014": "neu",
    "10.srt_0015": "neu",
    "10.srt_0016": "neu",
    "10.srt_0018": "neu",
    "10.srt_0019": "neg",  # shortage of specialists
    "10.srt_0020": "neu",
    "10.srt_0022": "neu",
    "10.srt_0023": "neu",
    "10.srt_0024": "neu",
    "10.srt_0026": "neu",
    "10.srt_0027": "neu",
    "10.srt_0028": "neu",
    "10.srt_0030": "neu",
    "10.srt_0031": "neu",
    "10.srt_0032": "neu",
    "10.srt_0034": "neu",
    "10.srt_0035": "neg",  # health burden
    "10.srt_0036": "neg",  # insufficient medical staff
    "10.srt_0038": "neg",  # hospital crisis
    "10.srt_0039": "neg",  # access denied to care
    "10.srt_0040": "neg",  # lack of medicines
    "10.srt_0042": "neg",  # patient suffering
    "10.srt_0043": "neg",  # structural healthcare failure
    "10.srt_0044": "pos",  # reform/improvement announced
    "10.srt_0046": "neg",  # inequality in access
    "10.srt_0047": "neg",  # overloaded hospitals
    "10.srt_0048": "neg",  # waitlists / denial of service
    "10.srt_0050": "neg",  # lack of resources
    "10.srt_0051": "neg",  # budget insufficiency
    "10.srt_0052": "neu",
    "10.srt_0054": "neg",  # poor care quality
    "10.srt_0055": "neg",  # health inequality urban/rural
    "10.srt_0056": "neg",  # corruption in procurement
    "10.srt_0058": "neg",  # medication shortages
    "10.srt_0059": "neg",  # staff exodus
    "10.srt_0060": "neg",  # system dysfunction
    "10.srt_0062": "neg",  # systemic healthcare problem

    # ── 2.srt  (garbled ASR — mixed pharma/health/social topics, 46 utts) ───
    "2.srt_0022": "neu",
    "2.srt_0042": "neu",
    "2.srt_0062": "neu",
    "2.srt_0100": "neu",
    "2.srt_0131": "neu",
    "2.srt_0138": "neu",
    "2.srt_0156": "neu",
    "2.srt_0171": "neu",
    "2.srt_0212": "neu",
    "2.srt_0238": "neu",
    "2.srt_0248": "neg",  # trampling rights and law
    "2.srt_0257": "neu",
    "2.srt_0283": "neu",
    "2.srt_0308": "neu",
    "2.srt_0328": "neu",
    "2.srt_0350": "neu",
    "2.srt_0402": "neu",
    "2.srt_0424": "neg",  # medical errors in Morocco
    "2.srt_0434": "neu",
    "2.srt_0467": "neg",  # vitamin shortage
    "2.srt_0473": "neg",  # vitamins no longer available
    "2.srt_0477": "neg",  # pesticides in food
    "2.srt_0484": "neu",
    "2.srt_0497": "neu",
    "2.srt_0517": "neu",
    "2.srt_0533": "neu",
    "2.srt_0538": "neu",
    "2.srt_0547": "neg",  # temperatures rising (environmental negative)
    "2.srt_0556": "neu",
    "2.srt_0559": "neu",
    "2.srt_0569": "neu",
    "2.srt_0582": "neu",
    "2.srt_0593": "neu",
    "2.srt_0599": "neu",
    "2.srt_0607": "neu",
    "2.srt_0619": "neg",  # Herzl / wounded / conflict
    "2.srt_0632": "neu",
    "2.srt_0662": "neg",  # social and economic problems in the country
    "2.srt_0681": "neu",
    "2.srt_0700": "neg",  # smoking / artery diseases
    "2.srt_0708": "neu",
    "2.srt_0717": "neg",  # heart and artery disease acknowledgment
    "2.srt_0728": "neu",
    "2.srt_0742": "neu",
    "2.srt_0747": "pos",  # national rail smoking ban (public health improvement)
    "2.srt_0777": "neu",

    # ── 3.srt  (tax law / Finance Law 2025, 46 utts) ────────────────────────
    "3.srt_0002": "neg",  # e-commerce earners dodge taxes; unfair system
    "3.srt_0003": "neu",  # CIMR sponsorship intro
    "3.srt_0004": "pos",  # CIMR retirement benefit to 100% of income
    "3.srt_0005": "neu",  # Finance Law 2025 context
    "3.srt_0006": "neu",  # tax reform overview
    "3.srt_0007": "neu",  # direct vs indirect taxes definition
    "3.srt_0008": "neu",  # reform measures description
    "3.srt_0009": "neg",  # middle class got nothing from reforms
    "3.srt_0010": "neg",  # middle class / employees feel oppressed
    "3.srt_0011": "neg",  # equity principle violated in taxation
    "3.srt_0012": "neu",  # different income categories definition
    "3.srt_0013": "neu",  # income tax table explanation
    "3.srt_0014": "neu",  # auto-entrepreneur thresholds
    "3.srt_0015": "neu",  # income limits per category
    "3.srt_0016": "neu",  # company accounting system
    "3.srt_0017": "neu",  # property / agricultural tax rates
    "3.srt_0018": "neu",  # >1M DH: two accounting methods
    "3.srt_0019": "neg",  # double taxation: corporate + personal
    "3.srt_0020": "neu",  # 20% dividend tax / capital gains
    "3.srt_0021": "neg",  # same citizens, same income, different taxes — unfair
    "3.srt_0022": "neu",  # 61B DH income tax statistics
    "3.srt_0023": "neu",  # government officials' counter-argument
    "3.srt_0024": "neg",  # allowances benefiting some but not others
    "3.srt_0025": "neu",  # progressive tax rate explanation
    "3.srt_0026": "neu",  # 2025 reform: new thresholds
    "3.srt_0027": "pos",  # allowances increased, lighter for middle class
    "3.srt_0028": "pos",  # calculation showing benefit for employees
    "3.srt_0029": "neg",  # employees actually gained nothing in practice
    "3.srt_0030": "neu",  # progressive rate structure
    "3.srt_0031": "neg",  # self-employed pay less; unfair to employees
    "3.srt_0032": "neu",  # historical tax payment method
    "3.srt_0033": "neu",  # comprehensive tax audit procedure
    "3.srt_0034": "neu",  # who has a tax ID
    "3.srt_0035": "neg",  # tax authority interrogating informal earners
    "3.srt_0036": "neg",  # expanded fiscal tracking of informal earners
    "3.srt_0037": "neu",  # new "other income" 6th category definition
    "3.srt_0038": "neg",  # taxing previously untaxed foreign income
    "3.srt_0039": "neu",  # future implementation details pending
    "3.srt_0040": "neg",  # state unable to collect from foreign platforms
    "3.srt_0041": "neu",  # open question on distinguishing content creators
    "3.srt_0042": "neg",  # morally questionable new legal category
    "3.srt_0043": "neu",  # expropriation process definition
    "3.srt_0044": "neg",  # expropriation compensation now taxed
    "3.srt_0045": "neg",  # tax on harm/expropriation compensation
    "3.srt_0046": "neg",  # complicated change affecting vulnerable people
    "3.srt_0048": "pos",  # tax relief for auto-entrepreneurs / small businesses

    # ── 4.srt  (public procurement guide, 45 utts) ──────────────────────────
    "4.srt_0002": "neg",  # public contracts associated with wasta and corruption
    "4.srt_0003": "pos",  # info can change your life; opportunity framing
    "4.srt_0004": "neu",  # CIH Bank sponsorship
    "4.srt_0006": "pos",  # small businesses can participate in contracts
    "4.srt_0007": "neu",  # government needs various services
    "4.srt_0008": "neu",  # oversight bodies listed
    "4.srt_0010": "neu",  # 3 types of contracts definition
    "4.srt_0011": "neu",  # 4 questions in procurement definition
    "4.srt_0012": "neu",  # need domain knowledge
    "4.srt_0014": "neu",  # identify needs → contract type
    "4.srt_0015": "neu",  # school meals example
    "4.srt_0016": "neu",  # security guards example
    "4.srt_0017": "neu",  # framework contracts (tranches)
    "4.srt_0018": "neu",  # companies help solve infrastructure problems
    "4.srt_0019": "neu",  # media content contracts for children/youth
    "4.srt_0020": "neu",  # competitive dialogue for complex needs
    "4.srt_0021": "neu",  # spontaneous offer / compensation
    "4.srt_0022": "neu",  # competition principle
    "4.srt_0023": "neu",  # 7 types of contracts
    "4.srt_0024": "neu",  # documents required
    "4.srt_0026": "neg",  # only 3 telecom companies; limited competition
    "4.srt_0027": "neu",  # threshold: 1M DH services / 10M works
    "4.srt_0028": "pos",  # Moroccan company gets pricing preference
    "4.srt_0029": "neu",  # shortened deadline regulatory change
    "4.srt_0030": "neu",  # negotiation procedure
    "4.srt_0031": "neu",  # finding tender listings
    "4.srt_0032": "neu",  # trade registration required
    "4.srt_0033": "neu",  # documents proving bidder identity
    "4.srt_0034": "neu",  # bid opening process
    "4.srt_0035": "neu",  # 3% performance bond
    "4.srt_0036": "pos",  # electronic procurement modernization
    "4.srt_0037": "neg",  # disqualification for procedural errors (risk)
    "4.srt_0038": "pos",  # can challenge if process violated (rights protection)
    "4.srt_0039": "pos",  # documents available publicly online (transparency)
    "4.srt_0040": "neu",  # challenge specs procedure
    "4.srt_0041": "neu",  # clean dossier → financial envelope
    "4.srt_0042": "neu",  # average price calculation
    "4.srt_0043": "pos",  # local company 15% preference
    "4.srt_0044": "pos",  # price visible to all after award (transparency)
    "4.srt_0046": "neu",  # Jeid Fournisseur payment portal
    "4.srt_0047": "neu",  # procurement is main (not only) way state buys
    "4.srt_0048": "pos",  # 1000 DH/year opportunity for small suppliers
    "4.srt_0050": "neu",  # need good reputation and references
    "4.srt_0051": "neu",  # past client experience matters
    "4.srt_0052": "neu",  # subscribe/share meta

    # ── 5.srt  (Casablanca stock exchange explainer, 46 utts) ───────────────
    "5.srt_0001": "pos",  # TGCC shares multiplied in value
    "5.srt_0002": "neg",  # stock investment labelled as gambling; fear
    "5.srt_0003": "neu",  # inflation erodes savings (intro framing)
    "5.srt_0004": "neu",  # educational intro
    "5.srt_0005": "neu",  # Dutch East India Company history
    "5.srt_0006": "neu",  # shares sold at 100 each (historical)
    "5.srt_0007": "neu",  # shares traded between people
    "5.srt_0008": "neu",  # Amsterdam / origin of "Bourse"
    "5.srt_0009": "neu",  # bonds: government/company debt definition
    "5.srt_0010": "neu",  # 500 DH bond at 5% interest example
    "5.srt_0011": "neu",  # raw materials market definition
    "5.srt_0012": "neu",  # supply/demand determines price
    "5.srt_0013": "neu",  # farmer futures contracts example
    "5.srt_0014": "neu",  # coffee futures example
    "5.srt_0015": "neg",  # speculator may lose if price falls
    "5.srt_0016": "neu",  # currency markets definition
    "5.srt_0017": "neg",  # crypto: high volatility / risk
    "5.srt_0018": "neu",  # market indices / MASI definition
    "5.srt_0019": "neu",  # index measures energy / top companies
    "5.srt_0020": "neu",  # investment funds: diversify risk
    "5.srt_0021": "neg",  # Casablanca exchange tiny; only 75 companies
    "5.srt_0022": "neg",  # Moroccan law restricts foreign investment
    "5.srt_0023": "neg",  # companies avoid listing; prefer opacity
    "5.srt_0024": "neu",  # can sell shares outside exchange
    "5.srt_0025": "neu",  # valuation based on assets/contracts
    "5.srt_0026": "neu",  # off-market sale needs notary
    "5.srt_0027": "neu",  # management structure requirements
    "5.srt_0029": "neu",  # analyst looks at clients, credits, outlook
    "5.srt_0030": "neu",  # company valuation 2B DH example
    "5.srt_0031": "neu",  # IPO: existing vs new shares definition
    "5.srt_0032": "neu",  # secondary IPO: investors sell existing shares
    "5.srt_0033": "neu",  # financial intermediary required
    "5.srt_0034": "neu",  # subscription period procedure
    "5.srt_0035": "neu",  # full allocation at fixed IPO price
    "5.srt_0036": "neg",  # overpriced IPO → value will fall
    "5.srt_0037": "neu",  # capital increase explanation
    "5.srt_0038": "neu",  # new shares added to pool
    "5.srt_0039": "neu",  # proceeds must fund announced purpose
    "5.srt_0040": "neu",  # pre-IPO institutional investors only
    "5.srt_0041": "neu",  # Cash Plus public offering Nov 2024
    "5.srt_0042": "pos",  # long-term investment in good company: shares grow
    "5.srt_0043": "neg",  # AI company stock overvaluation risk
    "5.srt_0044": "neg",  # financial crises; people fear stock market
    "5.srt_0045": "neu",  # broker advises and executes trades
    "5.srt_0046": "neu",  # online broker account setup
    "5.srt_0047": "neu",  # Maroclear clearing house

    # ── 6.srt  (Sufi/religious content, heavily garbled ASR, 45 utts) ───────
    "6.srt_0001": "neu",
    "6.srt_0019": "neu",
    "6.srt_0036": "neu",
    "6.srt_0061": "neu",
    "6.srt_0064": "neu",
    "6.srt_0079": "neu",
    "6.srt_0100": "neg",  # signs of moral failure / immorality in our society
    "6.srt_0103": "neg",  # victims forced to do things against their will
    "6.srt_0119": "neu",
    "6.srt_0150": "neu",
    "6.srt_0209": "neu",
    "6.srt_0218": "neu",
    "6.srt_0227": "neu",
    "6.srt_0231": "neu",
    "6.srt_0235": "neu",
    "6.srt_0243": "neu",
    "6.srt_0257": "neu",
    "6.srt_0264": "neu",
    "6.srt_0268": "neu",
    "6.srt_0272": "neu",
    "6.srt_0297": "neu",
    "6.srt_0308": "neu",
    "6.srt_0310": "neu",
    "6.srt_0321": "neu",
    "6.srt_0328": "neu",
    "6.srt_0342": "neu",
    "6.srt_0363": "neu",
    "6.srt_0369": "neu",
    "6.srt_0382": "neu",
    "6.srt_0392": "neu",
    "6.srt_0399": "neu",
    "6.srt_0402": "neu",
    "6.srt_0413": "neu",
    "6.srt_0421": "neu",
    "6.srt_0428": "neu",
    "6.srt_0436": "neu",
    "6.srt_0440": "neu",
    "6.srt_0447": "neu",
    "6.srt_0452": "neu",
    "6.srt_0464": "neu",
    "6.srt_0469": "neu",
    "6.srt_0499": "neu",
    "6.srt_0540": "neu",
    "6.srt_0573": "neu",
    "6.srt_0635": "neu",

    # ── 7.srt  (Morocco / Western Sahara history with historian Nabil Mouline,
    #           45 utts) ───────────────────────────────────────────────────────
    "7.srt_0002": "neu",  # intro with historian
    "7.srt_0003": "neu",  # Cash Plus sponsorship
    "7.srt_0004": "neu",  # investment disclaimer
    "7.srt_0005": "neg",  # 11th-c. tensions, revolutions in Mediterranean world
    "7.srt_0006": "pos",  # first 100% Moroccan empire: Almoravid
    "7.srt_0007": "neg",  # balance collapsed; internal divisions + foreign threats
    "7.srt_0008": "pos",  # Battle of Oued Al-Makhazin 1578; Sahara = key to power
    "7.srt_0009": "pos",  # Sahara recognized as strategic depth, not a margin
    "7.srt_0010": "neg",  # limited control over distant provinces; weakness
    "7.srt_0011": "neu",  # governance through tribes and arbitration dahirs
    "7.srt_0012": "neg",  # France entering Algeria: beginning of colonial era
    "7.srt_0013": "neg",  # European colonization of Morocco predates 1912
    "7.srt_0014": "neg",  # new colonizer France threatens Morocco
    "7.srt_0015": "neg",  # French colonial terms invented to justify expansion
    "7.srt_0016": "neg",  # France occupies El-Oued 1852; eyes southern Morocco
    "7.srt_0017": "neg",  # 1884: Europeans to divide Western Sahara (catastrophe)
    "7.srt_0018": "neg",  # Protectorate 1912; resistance continues to 1919
    "7.srt_0019": "neg",  # international powers controlled Sahara 1900-1934
    "7.srt_0020": "pos",  # Moroccan identity/sovereignty expressed through texts; 1956 independence
    "7.srt_0021": "neg",  # Spain's secret plan: artificial republic Reguibat
    "7.srt_0022": "neg",  # Operation Ecouvillon destroyed opposition; post-independence tensions
    "7.srt_0023": "neg",  # border dispute Morocco–Algeria; Algeria rejects borders
    "7.srt_0024": "neg",  # Spain refuses UN resolution 2072 on Sahara
    "7.srt_0025": "neg",  # Morocco forced to cede Eastern Sahara in exchange
    "7.srt_0026": "neg",  # Polisario Front created; conflict intensifies
    "7.srt_0027": "neg",  # Spain threatens self-determination referendum 1974
    "7.srt_0028": "neu",  # self-determination principle definition
    "7.srt_0029": "pos",  # Green March: historic peaceful crossing (faith, patriotism)
    "7.srt_0030": "pos",  # Sahara returned to Morocco and Mauritania
    "7.srt_0031": "neg",  # Polisario moves freely; backed by Algeria and Libya
    "7.srt_0032": "neg",  # war costly for Morocco; heavy losses
    "7.srt_0033": "neg",  # strategic mistakes and setbacks; Morocco strained
    "7.srt_0034": "pos",  # berm strategy pushes Polisario back; Morocco gains
    "7.srt_0035": "neg",  # referendum idea failed before it started
    "7.srt_0036": "neg",  # bureaucratic labyrinth on voter eligibility
    "7.srt_0037": "neg",  # UN envoy Baker resigns; referendum impossible
    "7.srt_0038": "pos",  # Morocco operates on ground, not just rhetoric
    "7.srt_0039": "neg",  # governance/participation problem; Hirak protests
    "7.srt_0040": "pos",  # UN Security Council calls Morocco's plan "serious and realistic"
    "7.srt_0041": "pos",  # Morocco doesn't accept deadlock; acts from 2012
    "7.srt_0042": "pos",  # African investment + religious soft power growing
    "7.srt_0043": "pos",  # Trump recognizes Moroccan Sahara; strengthens position
    "7.srt_0044": "pos",  # balance of power shifted definitively in Morocco's favor
    "7.srt_0045": "pos",  # Morocco's clear strategic weight; Spain gets the message
    "7.srt_0046": "pos",  # France and Spain recognize Moroccan Sahara; colonial page closed

    # ── 7769.srt  (political debate — PJD on censure motion, 45 utts) ────────
    "7769.srt_0001": "neu",  # show intro
    "7769.srt_0002": "neu",  # topics / guest intro
    "7769.srt_0003": "neu",  # show description
    "7769.srt_0005": "neu",  # thanks / welcome
    "7769.srt_0006": "neu",  # welcome to guest
    "7769.srt_0007": "neu",  # censure motion context
    "7769.srt_0008": "neu",  # opposition coalition context
    "7769.srt_0009": "neu",  # "Bismillah" + censure motion definition
    "7769.srt_0010": "neg",  # censure: debate government performance; goal: withdraw confidence
    "7769.srt_0011": "neu",  # PJD explains position
    "7769.srt_0012": "neu",  # question about widening rift
    "7769.srt_0013": "neu",  # historical censure motions 1964 / 1990
    "7769.srt_0014": "neu",  # PJD: "waters have separated"
    "7769.srt_0015": "neu",  # accusations of opacity
    "7769.srt_0016": "neu",  # dispute over "first party" in opposition
    "7769.srt_0017": "neu",  # PJD explaining position
    "7769.srt_0018": "neu",  # heated argument / interruption
    "7769.srt_0019": "neg",  # RNI importers with party connections; refuse inquiry
    "7769.srt_0020": "neg",  # RNI refuses facts commission
    "7769.srt_0021": "neu",  # question about government's half-term record
    "7769.srt_0022": "neu",  # government can't achieve 4% growth
    "7769.srt_0024": "neg",  # 4% growth "absolutely impossible"
    "7769.srt_0025": "neg",  # government handled this the wrong way
    "7769.srt_0026": "neu",  # 17–18M in RAMED; subsidy context
    "7769.srt_0027": "neu",  # agricultural middle class description
    "7769.srt_0028": "neu",  # 70% small farmers description
    "7769.srt_0029": "neg",  # growth rate impossible; 1M jobs commitment unmet
    "7769.srt_0030": "neg",  # PM excluded 1.5M children from school bag grants
    "7769.srt_0031": "neg",  # ideological bias toward large investors; small ones ignored
    "7769.srt_0032": "neg",  # 13B DH misallocated (opposition AND government reports)
    "7769.srt_0033": "neg",  # imports subsidized but retail prices didn't drop
    "7769.srt_0036": "neg",  # prices only dropped after King's decree; government failed before
    "7769.srt_0037": "neg",  # import exemptions pocketed by importers
    "7769.srt_0038": "neg",  # 20B DH wasted if direct imports had stopped
    "7769.srt_0039": "neg",  # importers set own prices despite exemptions
    "7769.srt_0041": "neg",  # Ministry of Agriculture deflected responsibility
    "7769.srt_0042": "neg",  # subsidy distribution: partisan/electoral manipulation
    "7769.srt_0043": "neu",  # rhetorical question about political exploitation
    "7769.srt_0044": "neu",  # budget/economic numbers
    "7769.srt_0045": "neu",  # PJD government ministry experience (historical)
    "7769.srt_0046": "neg",  # "partisan manipulation; the man understands nothing" (insults)
    "7769.srt_0047": "neg",  # government has anti-corruption problem from the start
    "7769.srt_0049": "neu",  # question about PM's ability to gather civil society
    "7769.srt_0050": "neg",  # pension reform: PM admits conflict of interest in parliament
    "7769.srt_0051": "neg",  # national pension committee exists but no progress

    # ── 7770.srt  (political debate — USFP / Socialist withdrawal from censure
    #               motion, 45 utts) ──────────────────────────────────────────
    "7770.srt_0001": "neu",  # show intro
    "7770.srt_0002": "neu",  # intro topics
    "7770.srt_0003": "neu",  # clarifying tonight's discussion
    "7770.srt_0004": "neu",  # guest: socialist opposition leader
    "7770.srt_0006": "neu",  # thanks for invitation
    "7770.srt_0007": "neu",  # welcome
    "7770.srt_0008": "neu",  # welcome
    "7770.srt_0009": "neu",  # censure motion caused political debate
    "7770.srt_0010": "neg",  # PJD rejected it; other parties followed (breakdown)
    "7770.srt_0011": "neg",  # parliament session problems with majority
    "7770.srt_0012": "neg",  # Socialist party didn't sign facts commission; why?
    "7770.srt_0013": "neg",  # government consulted without enough time (72 hours)
    "7770.srt_0015": "neg",  # facts commission stalled; political impasse
    "7770.srt_0016": "neu",  # reference to prior episode
    "7770.srt_0017": "neu",  # question about what the small party wanted
    "7770.srt_0018": "neg",  # process became contested after drawing lots
    "7770.srt_0019": "neg",  # left session without result
    "7770.srt_0020": "neg",  # conflicting accounts of meetings
    "7770.srt_0021": "pos",  # an idea emerged and was good
    "7770.srt_0022": "neg",  # communication breakdown; they didn't respond
    "7770.srt_0023": "neu",  # institutional balance explanation
    "7770.srt_0024": "neu",  # 1990 comparison
    "7770.srt_0026": "neu",  # question about whether decision was individual or party
    "7770.srt_0027": "neu",  # Sunday meeting not final
    "7770.srt_0029": "neg",  # "they acted Thursday while saying wait for Friday" (bad faith)
    "7770.srt_0030": "neu",  # no final decision in any meeting
    "7770.srt_0031": "neu",  # procedural clarification on party status
    "7770.srt_0032": "neg",  # Socialist party suspends all coordination
    "7770.srt_0033": "neu",  # PJD / Mouvement Populaire leaders today
    "7770.srt_0035": "neg",  # "you hollowed out the initiative by waiting"
    "7770.srt_0036": "neg",  # PJD wants everything their way; drawing lots dispute
    "7770.srt_0038": "neg",  # calling PJD "first party": nothing to hope for
    "7770.srt_0039": "neu",  # need joint responses on certain topics
    "7770.srt_0040": "neg",  # censure motion given for wrong purposes; institutions weakened
    "7770.srt_0041": "neu",  # relationship with PJD is about ideas, not partnership
    "7770.srt_0042": "neg",  # PJD opportunism accusation (didn't join last year, now wants to)
    "7770.srt_0044": "neu",  # France parliament comparison
    "7770.srt_0045": "neg",  # opposition reached high level of fracture / internal conflict
    "7770.srt_0046": "neu",  # analytical observation on party identity
    "7770.srt_0048": "neg",  # opportunities existed but were lost
    "7770.srt_0049": "neg",  # PM can't admit conflict of interest; opposition lacks impact
    "7770.srt_0050": "neg",  # losing censure motion initiative = opposition weak?
    "7770.srt_0051": "neu",  # 4287 written questions filed (statistic)
    "7770.srt_0052": "neg",  # 1200 unanswered questions by government
    "7770.srt_0053": "neu",  # opposition common experience despite disputes

    # ── 8.srt  (corruption explainer, 41 utts) ──────────────────────────────
    "8.srt_0001": "neg",  # 115B DH embezzled from social security 1972–1992
    "8.srt_0002": "neg",  # master's degree sold; bribery; conflict of interest
    "8.srt_0003": "neg",  # Morocco CPI 37/100; dropped 6 pts; Minister says "unfightable"
    "8.srt_0004": "neu",  # crowdfunding / understanding corruption framing
    "8.srt_0005": "neu",  # crowdfunding appeal meta
    "8.srt_0006": "neg",  # those with power prioritize personal interest
    "8.srt_0007": "neg",  # patronage: chose nephew, not best candidate
    "8.srt_0009": "neu",  # Denmark 90/100 CPI comparison (neutral data point)
    "8.srt_0010": "neg",  # 10 years of corruption; no recognition of competence
    "8.srt_0011": "neg",  # honest person pushed out of system
    "8.srt_0012": "neg",  # empty work; no marriage prospects; life ruined by corruption
    "8.srt_0013": "neg",  # ICOR framing: corruption wastes investment efficiency
    "8.srt_0014": "neg",  # South Korea ICOR 2.9 vs Morocco: Morocco lags far behind
    "8.srt_0015": "neu",  # legal definition of corruption (ICPC, 2022)
    "8.srt_0016": "neu",  # legal changes: articles in criminal code
    "8.srt_0017": "neg",  # anti-corruption strategy 2015 — launched but failed
    "8.srt_0018": "neg",  # ICPC strategy report: multiple reasons it failed
    "8.srt_0019": "neu",  # crowdfunding meta
    "8.srt_0020": "neg",  # corruption manifests as unexplained/illicit wealth
    "8.srt_0021": "neu",  # officials must declare assets (procedural)
    "8.srt_0022": "neg",  # only 15 auditors for thousands of declarations (system weakness)
    "8.srt_0023": "neg",  # Morocco: only 15 auditors vs France's digital system
    "8.srt_0024": "neg",  # conflict of interest article incomplete; legal gap
    "8.srt_0025": "neu",  # presumption of innocence principle
    "8.srt_0026": "neg",  # Justice Minister ignores accountability principle
    "8.srt_0027": "neg",  # accountability principle exists but ignored
    "8.srt_0028": "neg",  # corruption widespread; need alternative approach
    "8.srt_0029": "neg",  # Morocco needs "illicit enrichment" offense
    "8.srt_0030": "neg",  # 2023 anti-corruption bill shelved by government
    "8.srt_0031": "neg",  # conflict of interest: relatives in decisions
    "8.srt_0032": "neg",  # pre-2022: corruption complaints only through prosecution
    "8.srt_0033": "neg",  # to report corruption you need concrete evidence (barrier)
    "8.srt_0034": "neg",  # 30,000+ complaints in 2019; whistleblower law still pending
    "8.srt_0035": "neg",  # Morocco lacks US-style whistleblower rewards
    "8.srt_0036": "neg",  # legislative process stalled; can't pass parliament
    "8.srt_0037": "neg",  # procedural laws enabling corruption; opposite direction
    "8.srt_0038": "neg",  # statistics prove no political will to fight corruption
    "8.srt_0039": "neg",  # brain drain: qualified people emigrate; feel disrespected
    "8.srt_0040": "neg",  # societal problem internalized; most people accept corruption
    "8.srt_0041": "neg",  # meritocracy vs nepotism: meritocracy not applied
    "8.srt_0042": "neg",  # forced to be corrupt just to get one's due rights

    # ── 9.srt  (Morocco / Maghreb / Sahara history series, 44 utts) ──────────
    "9.srt_0002": "neu",  # historical sources (coins, manuscripts)
    "9.srt_0003": "neg",  # Algerians / Moroccans: brotherhood strained by historical events
    "9.srt_0004": "neu",  # North Africa name origins (Maghreb, Mauritania)
    "9.srt_0006": "neu",  # Mauretania kingdom extent
    "9.srt_0007": "neg",  # 146 BCE Hannibal; Roman alliance refused; instability
    "9.srt_0008": "neg",  # Roman refusal to unify Mauretania; interference
    "9.srt_0009": "neu",  # Mauretania extends to Atlantic
    "9.srt_0010": "neg",  # documentation lost; gap in historical record
    "9.srt_0011": "neu",  # Romans only occupied Ceuta and Tangier
    "9.srt_0012": "neg",  # lack of historical documentation; lost knowledge
    "9.srt_0013": "pos",  # Tariq ibn Ziyad's famous quote / celebrated speech
    "9.srt_0014": "neu",  # south hard to tax; nomadic populations
    "9.srt_0015": "neu",  # traditional state: tribal confederation structure
    "9.srt_0016": "neu",  # traditional state army composition
    "9.srt_0017": "neg",  # Umayyads discriminated against indigenous converts
    "9.srt_0018": "neu",  # Idrisids in Fez; local dynasties
    "9.srt_0019": "neu",  # Amazigh dynasties
    "9.srt_0020": "neu",  # crowdfunding meta
    "9.srt_0022": "pos",  # Ibn Tumart: educator, mujahideen, new ideas
    "9.srt_0023": "neu",  # Banu Hammad dynasty 1040–1152
    "9.srt_0024": "neu",  # Muhammad ibn Tumart: Almohad creed
    "9.srt_0026": "neg",  # plague; dynasty weakening; internal rebellion
    "9.srt_0027": "neg",  # Wattasids hold capital; chaos elsewhere
    "9.srt_0028": "pos",  # Muslim populations call for help; solidarity response
    "9.srt_0029": "neu",  # European attack threat; appeal to Muslims
    "9.srt_0030": "neu",  # Saadians aim to build empire like Almohads
    "9.srt_0031": "pos",  # Saadian Sultan captures Ottoman governor in 6 months
    "9.srt_0032": "neg",  # Ottoman entry; imposing customs; locals upset
    "9.srt_0034": "pos",  # Moroccans united; forgot differences to defend sovereignty
    "9.srt_0035": "neg",  # 1576: pretender asks Ottomans for army (foreign intervention)
    "9.srt_0036": "neg",  # Ottomans try to dominate Morocco again
    "9.srt_0038": "pos",  # population calls on Ottomans to help against Portuguese
    "9.srt_0039": "pos",  # only option: war and jihad; Moroccans won
    "9.srt_0040": "neu",  # Portugal under Spain; succession
    "9.srt_0042": "pos",  # southward policy blocks Ottomans
    "9.srt_0043": "neu",  # Bay'a, tribute, Islamic law; consolidation
    "9.srt_0044": "pos",  # negotiated peace with Ottomans; borders recognized
    "9.srt_0045": "neg",  # Europeans lose interest after Americas; Spain takes Larache
    "9.srt_0046": "neu",  # Alawi dynasty from 1472; Moulay Rashid unifies
    "9.srt_0047": "neu",  # Ottoman autonomy system in provinces
    "9.srt_0048": "neu",  # 1800–1802: Sultan reasserts influence in southeast
    "9.srt_0049": "neg",  # ups and downs until 1830; French conquest of Algeria
    "9.srt_0050": "neu",  # subscribe to channel meta
    "9.srt_0051": "neu",  # outro
}


def main() -> None:
    rows = [json.loads(line) for line in open(JSONL_PATH)]

    # Validate all IDs are covered
    missing = [r["id"] for r in rows if r["id"] not in LABELS]
    extra = [k for k in LABELS if k not in {r["id"] for r in rows}]
    if missing:
        print(f"WARNING: {len(missing)} utterances not in LABELS dict: {missing[:10]}")
    if extra:
        print(f"WARNING: {len(extra)} LABELS keys not in JSONL: {extra[:10]}")

    records = []
    for r in rows:
        label = LABELS.get(r["id"])
        if label is None:
            continue
        records.append({
            "utterance_id": r["id"],
            "file": r["file"],
            "fmt": "srt",
            "detected_lang": "ar",
            "text": r["text"],
            "label": label,
        })

    df = pd.DataFrame(records)
    label_counts = df["label"].value_counts()
    print(f"Wrote {len(df)} rows → {OUT_PATH}")
    print(f"  neg={label_counts.get('neg',0)}  pos={label_counts.get('pos',0)}  neu={label_counts.get('neu',0)}")

    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    df.to_csv(OUT_PATH, index=False)


if __name__ == "__main__":
    main()
