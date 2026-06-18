# Protocole d'annotation — Jeu de test broadcast HACA v4 (gold multi-annotateur)

**Objet :** construire un jeu d'évaluation broadcast assez **grand**, **équilibré** et
**fiable** (multi-annotateur) pour mesurer les trois classes de *content-valence* — surtout
la classe **positive**, aujourd'hui non mesurable.
**Schéma d'étiquetage :** Rubrique v3 (content-valence) — voir
[ANNOTATION_RUBRIC_V3.md](ANNOTATION_RUBRIC_V3.md).
**Statut :** méthodologie pour la prochaine itération (suite à FINDINGS §8).

---

## 1. Pourquoi ce protocole (rappel du problème)

FINDINGS §8 a établi, preuves à l'appui, que le jeu actuel `domaine_reel_v2` **ne permet pas
de mesurer la classe positive** :
- 20 positifs seulement, **un seul annotateur**, pas de score d'accord ;
- en ré-étiquetant les mêmes 194 énoncés sous une rubrique cohérente (v3), **kappa = 0.784**
  et la classe positive bouge de **20 → 16** ;
- tous les modèles plafonnent à ~0.45–0.52 et la F1 positive oscille entre 0.08 et 0.32 —
  bruit pur à ce niveau d'effectif.

Le verrou n'est ni le modèle, ni les données d'entraînement : c'est **la taille et la
représentativité du jeu d'évaluation**. Ce protocole le corrige.

---

## 2. Cibles quantitatives

| Paramètre | Cible | Justification |
|---|---|---|
| Positifs (gold) | **≥ 100** (min. absolu 60) | IC bootstrap à ±0.10 sur la F1 pos exige ~100 ex. (erreur-type du rappel ≈ √(p(1-p)/n) ; n≈100 → ±0.10 autour de 0.5) |
| Négatifs | ≥ 150 | mesure stable de neg |
| Neutres | ≥ 200 | classe majoritaire ; reflète le flux réel |
| Taille totale | **500–800** énoncés | compromis effort / fiabilité |
| Accord inter-annotateur | **kappa de Cohen ≥ 0.75** (global), **≥ 0.65 sur pos** après pilote | seuil « substantiel » ; sinon réviser la rubrique |
| Annotateurs | **2 indépendants + 1 adjudicateur** | mesure l'IAA et résout les désaccords |

> **20 positifs ne pourront jamais donner une F1 fiable.** Si la cible de 100 positifs est
> inatteignable (corpus trop pauvre), réduire l'ambition : ne **pas** prétendre mesurer la
> classe pos, et la signaler explicitement comme indicative.

---

## 3. Constitution du corpus (le verrou réel)

Le corpus actuel (12 fichiers : politique, fiscalité, corruption, histoire) est **structurellement
pauvre en positif** (~3 % du pool propre). Aucun protocole ne fabrique des positifs absents de
la source. Par ordre de priorité :

**3.1 Élargir la source (levier n°1, recommandé).**
Acquérir **10–20 fichiers broadcast supplémentaires** dans des genres **porteurs de positif**,
absents du corpus actuel :
- sport (victoires, exploits), culture (distinctions, festivals), économie (croissance,
  investissements, exports), science/innovation, santé (réussites, nouvelles infrastructures),
  développement régional, diplomatie (reconnaissances, partenariats).

Ces genres relèvent du contenu réellement régulé par la HACA et règlent **deux** problèmes à la
fois : la rareté du positif **et** le biais « sources non vues » (le même corpus élargi sert
alors à l'entraînement *et* au test, avec une séparation propre).

**3.2 Échantillonnage à deux strates** (sur le corpus élargi) :
- **Strate A — aléatoire** (~60 % du jeu) : échantillon stratifié par fichier
  (`extract_utterances.stratified_sample`), qui donne la **prévalence réelle** et une mesure
  fiable de neg/neu.
- **Strate B — enrichie en positifs** (~40 %) : pré-filtrer les candidats positifs par un
  **screening indépendant du modèle sous test** — liste de mots-clés pos (نجاح، تضاعف،
  استفاد، فرصة، إنجاز، نصر…) **+** un second avis LLM — puis annoter en priorité ces candidats
  jusqu'à atteindre les 100 positifs.

**3.3 Filtre qualité ASR.** Passer tout le pool par `src/asr_quality.py` ; **exclure** les
fragments illisibles (ils gonflent artificiellement le neutre et bruitent la mesure). Conserver
éventuellement une petite strate « bruit ASR » étiquetée à part pour tester la robustesse.

**3.4 Anti-fuite + gel.** Filtrer tout candidat partageant un **5-gramme** avec le jeu
d'entraînement (`build_haca_pool.word_ngrams`) ; geler le jeu final avec un **SHA-256**.

---

## 4. Conventions d'étiquetage (cas-limites)

Rubrique v3 = on étiquette ce que le contenu **décrit**, pas le ton du présentateur. Au-delà,
fixer **explicitement** les frontières qui ont causé les désaccords v2/v3 (tous vers neutre) :

| Situation | Étiquette | Note |
|---|---|---|
| Réforme/mesure qui **bénéficie** à un groupe | `pos` | « ce qui est décrit est bon » |
| **Victoire / réalisation** (y compris historique) cadrée positivement | `pos` | appliquer de façon **cohérente** (ancien *et* moderne) |
| Croissance, opportunité offerte à l'auditeur | `pos` | |
| Échec, pénurie, corruption, perte, conflit | `neg` | |
| Un **service qui existe** / un **règlement** énoncé | **`neu`** | fait descriptif, pas un jugement de résultat |
| Mécanisme, définition, procédure, how-to | `neu` | même sur un sujet sensible |
| Narration historique sans jugement | `neu` | |
| Fragment ASR illisible | **exclu** (ou `neu`+`garbled`) | ne pas deviner |
| Énoncé mixte | clause **dominante** | la clause qui est le propos |

Fournir aux annotateurs un **jeu de calibration de 20–30 énoncés déjà tranchés** illustrant ces
règles (notamment les 22 cas qui ont basculé entre v2 et v3).

---

## 5. Procédure d'annotation

1. **Formation** : lecture de la rubrique v3 + jeu de calibration ; discussion des cas-limites.
2. **Pilote (50–100 énoncés)** : double annotation aveugle → calculer le **kappa**. Si < 0.65
   sur pos, **réviser la rubrique** (préciser les conventions) avant de continuer. Itérer.
3. **Annotation complète** : les 2 annotateurs étiquettent **indépendamment** et **à l'aveugle
   des prédictions des modèles** (pour éviter tout biais d'ancrage).
4. **Mesure de l'accord** : kappa de Cohen global **et par classe** (surtout pos) ; matrice
   d'accord.
5. **Adjudication** : tous les désaccords sont tranchés par un **3ᵉ annotateur** (ou discussion
   à trois) → étiquette finale. Consigner les motifs des cas difficiles.
6. **Gel & documentation** : `domaine_reel_v4.csv`, SHA-256, datasheet (provenance, strates,
   kappa, distribution), versionnement git.

---

## 6. Protocole d'évaluation (une fois le gold prêt)

- **Métrique principale** : macro-F1 + tableau **par classe avec support**.
- **Intervalle de confiance bootstrap** (1000 ré-échantillons) sur la **F1 positive** → toujours
  reporter l'IC, pas un point seul.
- **Reporter le kappa** du jeu à côté des scores des modèles (un modèle ne peut pas dépasser le
  plafond imposé par le bruit d'annotation).
- **Strate enrichie (B)** : reporter la **F1 par classe** uniquement (pas l'*accuracy* — la
  prévalence y est volontairement biaisée).
- **Strate aléatoire (A)** : prévalence réelle + matrice de confusion.
- Réutiliser le harness existant (`src/harness.py`) ; ajouter `domaine_reel_v4` aux
  `eval_langs` / `--test`.

---

## 7. Pièges à éviter

- **Annoter en voyant les sorties du modèle** → ancrage. Annotation à l'aveugle obligatoire.
- **Échantillonner la strate B avec le modèle sous test** → mesure auto-validante. Utiliser un
  pré-filtre indépendant (mots-clés + autre LLM).
- **Fuite train/test** → filtre 5-grammes + exclusion par fichier source si nécessaire.
- **Reporter l'accuracy sur la strate enrichie** → trompeur (prévalence artificielle).
- **Un seul annotateur** → c'est précisément ce qui a échoué ; l'IAA est non négociable.

---

## 8. Livrables

- `data/test_sets/domaine_reel_v4.csv` (gold gelé, SHA-256) + strate « bruit ASR » à part.
- `report/HACA_TEST_V4_DATASHEET.md` : provenance, genres, strates, distribution, **kappa
  global et par classe**, limites.
- Scripts : extraction (réutilise `build_haca_pool.py`), pré-filtre pos, calcul d'IAA,
  adjudication.
- Rapport d'accord inter-annotateur (matrice, kappa, cas adjugés).

---

## 9. Résumé en une phrase

Le seul chemin vers un score positif chiffrable est un **corpus élargi vers des genres porteurs
de positif**, échantillonné en deux strates (aléatoire + enrichie), étiqueté par **deux
annotateurs indépendants** sous la rubrique v3 puis **adjugé**, avec **≥ 100 positifs** et un
**kappa ≥ 0.75** — faute de quoi la classe positive doit être déclarée non mesurable.
