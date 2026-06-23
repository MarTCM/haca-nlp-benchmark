# domaine_reel_v4_balanced — Datasheet

**Type :** jeu d'évaluation **équilibré de diagnostic** (PAS un gold de référence).
**Taille :** 450 énoncés — **neg 150 / neu 200 / pos 100** (cibles du protocole atteintes).
**Fichier :** `data/test_sets/domaine_reel_v4_balanced.csv`
**SHA-256 :** `8ccc28e3656904322103f36d6937ba979c73b92948a375bbbcd94ada5676a541`
**Construit par :** `src/build_domaine_reel_v4.py` (+ `src/synthetic_haca_test.py`).

---

## 1. Pourquoi ce fichier existe
Application du [protocole](TEST_SET_PROTOCOL.md) **sur les 12 SRT disponibles**, pour obtenir un
jeu qui atteint les effectifs par classe du protocole (≥100 pos / ≥150 neg / ≥200 neu). Comme le
corpus réel ne contient que ~35 positifs au total, les minimums sont atteints par **complément
synthétique**. C'est donc un **diagnostic équilibré**, utile pour observer le comportement par
classe sans le biais de prévalence — **pas** une mesure de généralisation réelle.

## 2. Composition

| Classe | réel (human_gold) | synthétique | total |
|---|---|---|---|
| neg | 63 | 87 | 150 |
| neu | 111 | 89 | 200 |
| pos | 20 | 80 | 100 |
| **total** | **194** | **256** | **450** |

Colonne `source` : `human_gold` (réel) vs `synthetic`.

- **Réel** = les 194 énoncés humains de `domaine_reel_v2` — réellement tenus à l'écart de
  l'entraînement (jamais entraînés dessus, dé-fuités).
- **Synthétique** = énoncés frais rédigés par Claude (`synthetic_haca_test.py`), **distincts**
  du synthétique d'entraînement (0 doublon exact, 0 chevauchement 5-grammes avec le **réel**
  d'entraînement).

## 3. ⚠ Avertissement (à lire avant d'interpréter)
Les modèles ont été **entraînés sur du synthétique positif rédigé par Claude**. Les positifs
synthétiques de ce jeu mesurent donc la **mémorisation de style**, pas la généralisation : la
**F1 pos sur l'ensemble complet sera optimiste et trompeuse pour un usage déploiement**.

**Toujours reporter les deux :**
- **réel seul** (`source == human_gold`) = le vrai signal — **identique aux résultats déjà
  publiés sur `domaine_reel_v2`** (FINDINGS §8 : MARBERTv2-haca 0.459, etc.) ;
- **équilibré complet** = diagnostic uniquement (effet de la mémorisation du synthétique).

Le **gold indépendant reste `domaine_reel_v2`** (humain). `domaine_reel_v3` (rubrique cohérente,
Claude) reste l'artefact de cohérence. Ce v4 ne remplace ni l'un ni l'autre.

## 4. Comment évaluer
```bash
# encoder (défaut + seuils calibrés), sur l'ensemble équilibré
python src/calibrate_thresholds.py --model marbertv2-haca --test-csv data/test_sets/domaine_reel_v4_balanced.csv
# LLM avec rubrique
python src/eval_llm_rubric.py --model 2b --test data/test_sets/domaine_reel_v4_balanced.csv
```
Pour le **réel seul**, se référer aux résultats `*_domaine_reel_v2.json` déjà obtenus (le
sous-ensemble `human_gold` de v4 EST `domaine_reel_v2`).

## 5. Limites
- 57 % synthétique → **non représentatif** de la prévalence réelle (broadcast = majoritairement
  neutre, positif rare).
- Mono-annotateur (réel : humain ; synthétique : Claude) — pas d'IAA. Le protocole v4 « propre »
  exige toujours un corpus élargi + double annotation humaine + adjudication.
