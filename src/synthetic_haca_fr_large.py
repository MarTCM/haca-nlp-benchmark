"""
French HACA-Sent — LARGE synthetic broadcast-French set, authored by Claude.

NO LLM API. Every template, slot value and lexicon below is hand-written by Claude in this
project (per the project rule). Diversity and scale come from *combinatorial* composition of
those hand-authored parts — not from any external model. This is a deterministic generator
(seed 42), so the output is reproducible.

Why this exists: the first French fine-tune used only 143 clean synthetic sentences and lost
to the off-the-shelf model on the real gold (report/FINETUNING.md §6). Two problems:
  (1) too few examples, and
  (2) too clean — the real gold is noisy ASR (dropped accents, homophones, run-ons, [Musique]).
This file fixes both: ~N_PER_CLASS examples per class from many templates × rich slot banks,
and an ASR-noise augmentation pass that rewrites a fraction of rows to look like real subtitles.

Output: data/test_sets/synthetic_haca_fr_large.csv  (used by finetune.py if present).

Usage:
    python src/synthetic_haca_fr_large.py                 # defaults: 1500/class, 45% noised
    python src/synthetic_haca_fr_large.py --per-class 2000 --noise-frac 0.5
"""

import argparse
import csv
import os
import random
import re

OUT_CSV = "data/test_sets/synthetic_haca_fr_large.csv"
SEED = 42

# ── shared, hand-authored slot banks (broadcast / Moroccan-French register) ──────────────
REGION_IN = [
    "à Tanger", "à Casablanca", "à Agadir", "à Marrakech", "à Fès", "à Oujda", "à Laâyoune",
    "à Dakhla", "à Kénitra", "à Tétouan", "dans l'Oriental", "dans le Souss", "dans le Rif",
    "dans la région de Marrakech", "dans la région de l'Oriental", "dans plusieurs régions",
    "en milieu rural", "dans les zones enclavées", "dans le sud du pays", "dans le nord",
]
# domain noun-phrases usable as a subject (verb stays 3rd-person, no adjective agreement)
DOMAIN_NP = [
    "le secteur de la santé", "le système éducatif", "la filière agricole", "le secteur touristique",
    "l'industrie automobile", "l'industrie textile", "le secteur de la pêche", "le marché du logement",
    "le réseau de transport", "le secteur numérique", "le secteur de l'énergie", "le secteur du BTP",
    "la filière des phosphates", "le secteur de l'artisanat", "le secteur agroalimentaire",
    "le tissu des petites entreprises",
]
# "le secteur {SECTOR}" -> grammatical with baked contraction
SECTOR = [
    "de la santé", "de l'éducation", "du tourisme", "de l'agriculture", "de la pêche",
    "du logement", "des transports", "du numérique", "de l'énergie", "de l'automobile",
    "de l'agroalimentaire", "du textile", "de la construction", "de l'artisanat",
    "de la formation professionnelle", "de l'industrie",
]
GROUP_PL = [
    "les jeunes", "les agriculteurs", "les ouvriers", "les familles modestes", "les patients",
    "les habitants", "les petites entreprises", "les étudiants", "les fonctionnaires",
    "les diplômés", "les artisans", "les pêcheurs", "les commerçants", "les retraités",
    "les ménages", "les investisseurs", "les usagers",
]
NUM = ["des milliers de", "des centaines de", "plus de 10 000", "près de 5 000", "quelque 20 000",
       "de nombreux", "plusieurs centaines de", "des dizaines de milliers de"]
PCT = ["de 12 %", "de 20 %", "de près de 8 %", "de plus de 15 %", "de 5 points", "de 30 %",
       "de moitié", "d'un tiers"]
INTRO = ["", "", "Selon les chiffres officiels, ", "D'après le rapport, ", "Cette année, ",
         "Ces derniers mois, ", "Sur le terrain, ", "Dans les faits, ", "À en croire les autorités, ",
         "Comme le montre notre reportage, "]

# ── POSITIVE templates (content reports something good) ──────────────────────────────────
POS = [
    "{intro}les exportations {sector} ont atteint un niveau record cette année, créant {num} emplois.",
    "{intro}le secteur {sector} a connu une croissance {pct}, confortant la place du pays en Afrique.",
    "{intro}un nouvel hôpital a été ouvert {region_in} et desservira {num} habitants.",
    "{intro}{group} ont vu leur pouvoir d'achat s'améliorer après la dernière réforme.",
    "{intro}les investissements étrangers ont progressé {pct}, signe de la confiance des marchés.",
    "{intro}{domain} a créé {num} postes {region_in} grâce aux nouveaux projets.",
    "{intro}la réforme a permis à {num} familles d'accéder à une couverture médicale.",
    "{intro}le taux de chômage a reculé {pct} grâce aux programmes d'emploi des jeunes.",
    "{intro}une grande usine a ouvert ses portes {region_in} et emploiera {num} personnes.",
    "{intro}{group} bénéficient désormais d'un soutien direct qui a soulagé leur quotidien.",
    "{intro}le pays a inauguré une nouvelle ligne à grande vitesse qui a rapproché les villes.",
    "{intro}les bourses ont permis à {num} étudiants modestes de poursuivre leurs études.",
    "{intro}la production {sector} a fortement augmenté et a ouvert de nouveaux marchés à l'étranger.",
    "{intro}le tourisme a explosé {region_in}, avec une hausse {pct} des visiteurs.",
    "{intro}une équipe marocaine a réalisé une avancée saluée bien au-delà des frontières.",
    "{intro}le programme de logement a aidé {num} ménages à devenir propriétaires à un prix abordable.",
    "{intro}{domain} s'est modernisé {region_in} et offre désormais de meilleurs services.",
    "{intro}les énergies renouvelables fournissent une part croissante de l'électricité du pays.",
    "{intro}{group} ont salué la baisse des prix obtenue après les nouvelles mesures.",
    "{intro}la formation professionnelle a doté {num} jeunes de compétences recherchées.",
    "{intro}le port a battu un record de trafic et a renforcé les échanges {region_in}.",
    "{intro}une start-up a levé des fonds importants et va recruter {num} ingénieurs.",
    "{intro}la campagne agricole a été excellente et a donné une récolte abondante.",
    "{intro}le pays a signé un accord qui ouvrira de nouveaux débouchés aux produits locaux.",
    "{intro}les caravanes médicales ont soigné {num} patients {region_in} qui n'avaient pas accès aux soins.",
    "{intro}{group} profitent d'un nouveau dispositif qui finance leurs projets sans intérêts.",
    "{intro}une médaille a récompensé les sportifs du pays, à la grande fierté du public.",
    "{intro}l'électricité et l'eau potable atteignent désormais des villages longtemps oubliés {region_in}.",
]

# ── NEGATIVE templates (content reports something bad) ───────────────────────────────────
NEG = [
    "{intro}les prix des produits de base ont flambé et {group} peinent à joindre les deux bouts.",
    "{intro}la sécheresse a ruiné les récoltes {region_in} et aggravé la détresse du monde rural.",
    "{intro}l'hôpital manque de médecins et de matériel, et {group} attendent des mois un rendez-vous.",
    "{intro}le chômage des jeunes a atteint {pct} et {num} diplômés restent sans emploi.",
    "{intro}un nouveau scandale de corruption a englouti {num} dirhams d'argent public.",
    "{intro}une pénurie d'eau {region_in} a contraint {group} à réclamer des solutions d'urgence.",
    "{intro}{domain} souffre d'un sous-investissement chronique qui pénalise {group}.",
    "{intro}la hausse des carburants {pct} a accru la pression sur le budget des ménages.",
    "{intro}{group} ont manifesté {region_in} pour dénoncer la dégradation des services publics.",
    "{intro}le projet a pris des années de retard et les fonds se sont volatilisés sans résultat.",
    "{intro}des inondations ont emporté les routes {region_in} et isolé {num} habitants.",
    "{intro}l'émigration des médecins s'est accentuée, laissant {domain} exsangue.",
    "{intro}la pollution industrielle {region_in} a provoqué des maladies chez {group}.",
    "{intro}les loyers ont grimpé {pct} et empêchent {group} de se loger correctement.",
    "{intro}la fermeture d'une usine a laissé {num} ouvriers sans emploi {region_in}.",
    "{intro}{group} dénoncent une exploitation et des conditions de travail indignes.",
    "{intro}le taux de pauvreté reste élevé {region_in}, où {group} vivent sans revenu stable.",
    "{intro}l'évasion fiscale prive le Trésor de recettes destinées à {domain}.",
    "{intro}des coupures d'eau répétées {region_in} ont éprouvé {group} pendant l'été.",
    "{intro}l'abandon scolaire reste fort {region_in}, surtout chez les filles.",
    "{intro}la mauvaise gestion des déchets {region_in} a créé un problème sanitaire.",
    "{intro}{domain} accumule les déficits et menace {num} emplois.",
    "{intro}les listes d'attente {region_in} ont conduit des patients à mourir faute de soins.",
    "{intro}la répression s'est durcie et {group} disent vivre dans la peur.",
    "{intro}la concurrence déloyale a poussé {num} petits producteurs à la faillite.",
    "{intro}le réseau internet défaillant {region_in} prive {group} de l'enseignement à distance.",
    "{intro}une hausse {pct} des prix de l'immobilier a éloigné le rêve de propriété pour {group}.",
    "{intro}le retard des indemnités a aggravé la détresse des sinistrés {region_in}.",
]

# ── NEUTRAL templates (procedural / definitional / descriptive — no recoverable valence) ──
NEU = [
    "{intro}le présent reportage revient sur l'organisation {sector} et le rôle de ses institutions.",
    "{intro}la loi de finances fixe les recettes et les dépenses de l'État pour une année.",
    "{intro}la taxe sur la valeur ajoutée s'applique à la plupart des biens et services.",
    "{intro}un marché public passe par plusieurs étapes, de la définition des besoins à l'attribution.",
    "{intro}nous recevons aujourd'hui notre invité pour évoquer le dossier {sector}.",
    "{intro}le rapport présente d'abord le contexte, puis les chiffres, avant les recommandations.",
    "{intro}la Cour des comptes contrôle l'emploi des deniers publics et publie des rapports annuels.",
    "{intro}la régionalisation confère aux régions des compétences plus larges sur leurs affaires.",
    "{intro}le régime de retraite fonctionne sur le principe de la cotisation des affiliés.",
    "{intro}dans un instant, nous verrons comment se déroule la procédure {region_in}.",
    "{intro}le décret définit plusieurs types de marchés selon la nature des besoins.",
    "{intro}la séance s'est ouverte par la lecture de l'ordre du jour {region_in}.",
    "{intro}le produit intérieur brut mesure la valeur des biens et services produits par le pays.",
    "{intro}l'autorité de régulation veille au respect du cahier des charges par les opérateurs.",
    "{intro}le texte distingue plusieurs catégories de bénéficiaires et leurs conditions d'éligibilité.",
    "{intro}la commission a auditionné plusieurs responsables avant de rédiger ses conclusions.",
    "{intro}le recensement de la population est réalisé périodiquement pour actualiser les données.",
    "{intro}le contrat de travail précise la durée, la rémunération et les obligations de chacun.",
    "{intro}le document retrace les étapes du chantier, de l'appel d'offres à la réception.",
    "{intro}la séance de questions orales permet d'interroger les ministres sur leur politique.",
    "{intro}le panel réunit un économiste, un juriste et un représentant {sector}.",
    "{intro}le journaliste rappelle le calendrier des prochaines échéances {region_in}.",
    "{intro}la monnaie nationale est émise par la banque centrale qui veille à la stabilité des prix.",
    "{intro}le budget se divise en un volet de fonctionnement et un volet d'investissement.",
    "{intro}la déclaration de patrimoine est une procédure périodique pour les responsables publics.",
    "{intro}l'organisme assure la régulation {sector} et l'instruction des dossiers.",
    "{intro}le dispositif prévoit un guichet unique où le demandeur dépose et suit son dossier.",
    "{intro}le séminaire {region_in} a réuni des experts pour présenter l'état des lieux {sector}.",
]


def _fill(template: str, rng: random.Random) -> str:
    out = template.format(
        intro=rng.choice(INTRO), region_in=rng.choice(REGION_IN), sector=rng.choice(SECTOR),
        domain=rng.choice(DOMAIN_NP), group=rng.choice(GROUP_PL), num=rng.choice(NUM),
        pct=rng.choice(PCT),
    )
    out = " ".join(out.split())
    return out[0].upper() + out[1:] if out else out


# ── ASR-noise augmentation: make text look like real (imperfect) subtitle ASR ────────────
_ACCENTS = str.maketrans("àâäéèêëîïôöùûüç", "aaaeeeeiioouuuc")
_HOMO = [(" à ", " a "), (" a ", " à "), (" ces ", " ses "), (" ont ", " on "),
         (" et ", " est "), (" son ", " sont "), (" ce ", " se "), (" la ", " là ")]


def asr_noise(text: str, rng: random.Random) -> str:
    """Apply a random subset of ASR-style corruptions (deterministic via rng)."""
    t = text
    if rng.random() < 0.6:                       # drop accents (very common in ASR/SRT)
        t = t.translate(_ACCENTS)
    if rng.random() < 0.4:                       # one homophone confusion
        a, b = rng.choice(_HOMO)
        t = t.replace(a, b, 1)
    if rng.random() < 0.5:                       # drop final punctuation
        t = t.rstrip(".!?")
    if rng.random() < 0.3:                       # drop a random short word (mimic missed token)
        words = t.split()
        idx = [i for i, w in enumerate(words) if len(w) <= 3]
        if idx:
            del words[rng.choice(idx)]
            t = " ".join(words)
    if rng.random() < 0.15:                      # stray music/applause tag
        t += " [Musique]"
    if rng.random() < 0.2:                       # lowercase the start (no sentence capital)
        t = t[0].lower() + t[1:] if t else t
    if rng.random() < 0.15:                      # duplicate a word (ASR stutter)
        words = t.split()
        if len(words) > 3:
            i = rng.randrange(len(words))
            words.insert(i, words[i])
            t = " ".join(words)
    return " ".join(t.split())


def build_rows(per_class: int, noise_frac: float):
    rng = random.Random(SEED)
    rows, seen = [], set()
    uid = 0
    for templates, label in ((POS, "pos"), (NEG, "neg"), (NEU, "neu")):
        made, attempts = 0, 0
        while made < per_class and attempts < per_class * 60:
            attempts += 1
            text = _fill(rng.choice(templates), rng)
            noised = rng.random() < noise_frac
            if noised:
                text = asr_noise(text, rng)
            key = text.lower()
            if key in seen or len(text) < 30:
                continue
            seen.add(key)
            uid += 1
            made += 1
            rows.append({
                "utterance_id": f"synthfrL_{uid:05d}",
                "file": "synthetic_fr_large", "fmt": "synthetic",
                "detected_lang": "francais", "quality": "clean",
                "text": text, "label": label,
                "label_source": "claude-synth-template",
                "synthetic": True, "topic": "noised" if noised else "clean",
            })
    rng.shuffle(rows)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-class", type=int, default=1500)
    ap.add_argument("--noise-frac", type=float, default=0.45)
    ap.add_argument("--out", default=OUT_CSV)
    args = ap.parse_args()

    rows = build_rows(args.per_class, args.noise_frac)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    fields = ["utterance_id", "file", "fmt", "detected_lang", "quality", "text",
              "label", "label_source", "synthetic", "topic"]
    with open(args.out, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    from collections import Counter
    print(f"Wrote {len(rows)} rows → {args.out}")
    print(f"  label dist : {dict(Counter(r['label'] for r in rows))}")
    print(f"  noised     : {sum(r['topic'] == 'noised' for r in rows)} / {len(rows)}")


if __name__ == "__main__":
    main()
