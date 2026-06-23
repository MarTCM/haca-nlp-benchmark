# Benchmark Findings — Analyse des résultats

**Projet :** Analyse de sentiment multilingue pour la HACA
**Auteur :** Marwane ElBaraka
**Date :** Juin 2026
**Modèles évalués :** 9 (3 ready-made + 4 fine-tuned + 2 LLM zero-shot)
**Langues :** darija_ar · arabizi · francais · msa
**Jeux de test :** 1 000 utterances par langue (publics) + 194 utterances domaine réel

---

## 1. Vue d'ensemble des résultats

### Tableau complet — Macro-F1 par modèle × langue

| Modèle | Type | darija\_ar | arabizi | francais | msa | Moyenne |
|---|---|---|---|---|---|---|
| **DarijaBERT-arabizi** | fine-tuné | — | **0.983** | — | — | 0.983 |
| **distilcamembert** | ready-made | — | — | **0.949** | — | 0.949 |
| **camelbert-da** | ready-made | 0.701 | — | — | **0.924** | 0.812 |
| **MARBERTv2** | fine-tuné | **0.844** | — | — | 0.838 | 0.841 |
| **Atlas-Chat-2B** | LLM 4-bit | 0.727 | 0.978 | — | — | 0.852 |
| **QARIB** | fine-tuné | 0.827 | — | — | 0.812 | 0.819 |
| **Atlas-Chat-9B** | LLM 4-bit | 0.833 | 0.754† | — | — | 0.793 |
| **DarijaBERT** | fine-tuné | 0.775 | — | — | — | 0.775 |
| **xlm-t** | ready-made | 0.723 | 0.762 | 0.772 | 0.812 | 0.767 |

†  9B arabizi : 42,4 % des prédictions sont « neutre » sur un jeu de test binaire.

**Latences (ms/utterance) :**

| Modèle | darija\_ar | arabizi | francais | msa |
|---|---|---|---|---|
| MARBERTv2 fine-tuné | 2,5 | — | — | 4,4 |
| DarijaBERT-arabizi | — | 3,9 | — | — |
| QARIB | 2,7 | — | — | 4,5 |
| DarijaBERT | 2,7 | — | — | — |
| camelbert-da | 90,9 | — | — | 163,7 |
| xlm-t | 109,6 | 145,3 | 1 006,1 | 168,4 |
| distilcamembert | — | — | 420,9 | — |
| Atlas-Chat-2B (4-bit) | 186,4 | 184,5 | — | — |
| Atlas-Chat-9B (4-bit) | 451,0 | 451,3 | — | — |

---

## 2. Constatations par langue

### 2.1 Darija arabe (darija_ar)

Jeu de test : 1 000 utterances — neg=201, neu=232, pos=567 (tweets et commentaires YouTube marocains).

| Rang | Modèle | Macro-F1 |
|---|---|---|
| 1 | MARBERTv2 fine-tuné | **0.844** |
| 2 | Atlas-Chat-9B | 0.833 |
| 3 | QARIB fine-tuné | 0.827 |
| 4 | Atlas-Chat-2B | 0.727 |
| 5 | DarijaBERT fine-tuné | 0.775 |
| 6 | camelbert-da (ready-made) | 0.701 |
| 7 | xlm-t | 0.723 |

**Observations :**
- MARBERTv2 reste le meilleur modèle global sur la darija. Son avantage vient d'un F1 élevé sur la classe neg (0.854) et pos (0.927), grâce à un fine-tuning sur le jeu MAC (tweets marocains).
- Atlas-Chat-9B (0.833) dépasse QARIB (0.827) sans aucun fine-tuning, illustrant la force des LLM sur des langues sous-représentées en entraînement supervisé.
- camelbert-da (0.701), bien que conçu pour l'arabe dialectal, ne reçoit pas de fine-tuning ici et peine avec la classe neutre.

### 2.2 Arabizi (arabizi)

Jeu de test : 1 000 utterances binaires — neg=343, pos=657 (pas de classe neutre dans le jeu de test MYC).

| Rang | Modèle | Macro-F1 |
|---|---|---|
| 1 | DarijaBERT-arabizi fine-tuné | **0.983** |
| 2 | Atlas-Chat-2B | **0.978** |
| 3 | Atlas-Chat-9B | 0.754 |
| 4 | xlm-t | 0.762 |

**Observations :**
- DarijaBERT-arabizi (0.983) est le meilleur modèle toutes catégories, atteignant une quasi-perfection (F1 neg=0.978, F1 pos=0.987).
- Atlas-Chat-2B (0.978) est à 0.5 point près, malgré 10× plus de paramètres et 47× plus de latence — preuve que la taille ne garantit pas le gain.
- **L'effondrement du 9B (0.754) est le résultat le plus contre-intuitif du benchmark :** le modèle 9× plus grand est 22 points en dessous du 2B. Cause : 42,4 % des utterances reçoivent la prédiction « neutre » — classe absente du jeu de test binaire MYC. Le 9B est mieux calibré sur la nuance (il prédit « neutre » là où le 2B forcerait pos/neg), ce qui aide sur la darija 3 classes mais ruine les performances sur un jeu binaire. Ce phénomène est détaillé en §4.

### 2.3 Français (francais)

Jeu de test : 1 000 utterances — critiques de films Allociné (neg/pos/neu).

| Rang | Modèle | Macro-F1 |
|---|---|---|
| 1 | distilcamembert fine-tuné | **0.949** |
| 2 | xlm-t | 0.772 |

**Observations :**
- distilcamembert domine très nettement (+17,7 pts). C'est un modèle natif français : aucun gain n'est à attendre d'un fine-tuning supplémentaire.
- xlm-t est le seul autre modèle testé sur le français et reste correct mais loin derrière.

### 2.4 Arabe standard moderne (msa)

Jeu de test : 1 000 utterances ASTD (tweets politiques égyptiens).

| Rang | Modèle | Macro-F1 |
|---|---|---|
| 1 | camelbert-da (ready-made) | **0.924** |
| 2 | MARBERTv2 fine-tuné | 0.838 |
| 3 | QARIB fine-tuné | 0.812 |
| 4 | xlm-t | 0.812 |

**Observations :**
- camelbert-da (0.924) est étonnamment le meilleur, malgré l'absence de fine-tuning. Il a été pré-entraîné spécifiquement sur du texte arabe dialectal et MSA, ce qui lui confère un avantage natif sur ce jeu.
- MARBERTv2 fine-tuné sur la darija généralise bien au MSA (0.838), à seulement 8,6 pts du ready-made spécialisé.

---

## 3. Fine-tuning vs ready-made vs LLM

Le tableau ci-dessous résume la valeur ajoutée de chaque approche :

| Approche | Meilleur F1 darija | Meilleur F1 arabizi | Latence typ. | GPU requis |
|---|---|---|---|---|
| Fine-tuned encoder | **0.844** (MARBERTv2) | **0.983** (DarijaBERT-az) | ~3 ms | ≥ 4 Go VRAM |
| LLM zero-shot (2B) | 0.727 | 0.978 | ~185 ms | 4 Go (4-bit) |
| LLM zero-shot (9B) | 0.833 | 0.754 | ~451 ms | 6 Go (4-bit) |
| Ready-made encoder | 0.723 (xlm-t) | 0.762 (xlm-t) | ~100–1000 ms | CPU possible |

**Règle de décision (issue du PLAN) :**
> Préférer un encoder fine-tuné si son macro-F1 darija ≥ 0,75 **ou** si l'écart LLM–encoder < 5 points de F1.

Sur la darija : MARBERTv2 (0.844) dépasse le seuil de 0,75 et l'écart avec le meilleur LLM (9B, 0.833) est de 1,1 point — bien en dessous des 5 points. **L'encoder fine-tuné est recommandé.**

Sur l'arabizi : DarijaBERT-arabizi (0.983) dépasse le seuil de 0,75 et l'écart avec le LLM 2B (0.978) est de 0,5 point. **L'encoder fine-tuné est recommandé.**

---

## 4. L'inversion d'échelle : pourquoi le 9B est pire que le 2B sur l'arabizi

C'est la découverte la plus surprenante du benchmark.

| Modèle | darija\_ar (3 classes) | arabizi (binaire) |
|---|---|---|
| Atlas-Chat-2B | 0.727 | **0.978** |
| Atlas-Chat-9B | **0.833** | 0.754 |

Le 9B est meilleur sur la tâche à 3 classes et pire sur la tâche binaire. L'explication n'est pas un bug mais un **artefact de calibration** :

- Le 9B est mieux calibré que le 2B : il émet « neutre » quand il est incertain, là où le 2B force une prédiction pos/neg.
- Sur la darija (3 classes) : cette calibration aide. F1 neutre : 0.480 (2B) → 0.746 (9B), ce qui tire macro-F1 de 0.727 à 0.833.
- Sur l'arabizi (binaire, MYC) : il n'y a **pas de classe neutre dans le jeu de test**. Les 424/1 000 prédictions « neutre » du 9B comptent comme des erreurs. Recall pos s'effondre à 0.417.
- Le 2B, moins calibré mais plus direct, ne prédit presque jamais « neutre » sur ce jeu binaire — d'où son quasi-score parfait (0.978).

**Conclusion pratique :** pour un jeu de test binaire ou un contexte où la classe neutre est rare, le 2B est supérieur. Pour une tâche 3 classes sur de la darija, le 9B est le meilleur LLM testé.

---

## 5. L'écart domaine réel — la constatation principale pour la HACA

### 5.1 Protocole

194 utterances extraites de vraies émissions HACA et de vidéos YouTube marocaines (fichiers `.srt`), évaluées sur **trois jeux d'annotations distincts** pour mesurer la sensibilité des résultats à la philosophie d'annotation.

**Trois jeux d'annotations (194 utterances chacun) :**

| Jeu | Annotateur | Philosophie | neg | neu | pos |
|---|---|---|---|---|---|
| `domaine_reel.csv` | Humain (strict) | L'animateur exprime explicitement du sentiment | 38 | 146 | 10 |
| `gemini.csv` | Gemini 1.5 (large) | Le contenu décrit quelque chose de négatif/positif | 65 | 97 | 32 |
| `domaine_reel_v2.csv` | Humain (HACA) | Valence du contenu — ce qui est décrit importe, pas le ton | 63 | 111 | 20 |

La philosophie HACA retenue (**v2**) : une émission qui rapporte un scandale de corruption ou l'émigration des médecins est « négative », même si l'animateur reste neutre. C'est ce que la HACA régule — le **contenu diffusé**, pas la posture du présentateur.

- Langue : darija marocaine en script arabe (100%)
- Types de contenus : débats politiques, explications économiques/fiscales, histoire du Maroc, santé, anticorruption

Les 5 modèles Arabic testés en step 2–5 ont été évalués sur les trois jeux.

### 5.2 Résultats — comparaison multi-modèles et multi-annotations

**Macro-F1 sur les trois jeux d'annotations (même 194 utterances, philosophies différentes) :**

| Modèle | Public darija | Strict (v1) | Gemini (large) | **HACA (v2)** | Écart public→v2 |
|---|---|---|---|---|---|
| **xlm-t** | 0.723 | 0.413 | **0.613** | **0.489** | −0.234 |
| **DarijaBERT** | 0.775 | 0.439 | 0.526 | 0.415 | −0.360 |
| **QARIB** | 0.827 | 0.421 | 0.459 | 0.386 | −0.441 |
| **MARBERTv2** | **0.844** | **0.441** | 0.459 | 0.380 | −0.464 |
| **camelbert-da** | 0.701 | 0.306 | 0.513 | 0.374 | −0.327 |

**Observations clés :**
1. **xlm-t est le meilleur modèle sur les annotations broadened** (Gemini : 0.613, HACA v2 : 0.489), alors qu'il est quatrième sur les annotations strictes (0.413). Le classement s'inverse selon la philosophie d'annotation.
2. **L'écart de généralisation est plus petit pour xlm-t quel que soit le jeu** (−0.234 sur v2 vs −0.464 pour MARBERTv2). Les encoders fine-tunés sur MAC-tweets sur-apprennent des corrélations shortcut (argot Twitter, emojis) absentes du broadcast.
3. **Les scores Gemini sont artificiellement élevés** (0.459–0.613) car Gemini a annoté les mêmes types de contenus que ses propres sorties, créant un biais d'alignement. Les annotations HACA v2 (humain, indépendant) sont l'étalon fiable.
4. **camelbert-da présente un biais négatif extrême :** recall neg = 0.905 mais recall neu = 0.234 sur v2 — il sur-prédit neg massivement et ignore la majorité neutre.

### 5.3 Détail par classe (MARBERTv2 — modèle de référence)

| | Jeu public (MAC tweets) | Domaine réel HACA |
|---|---|---|
| **Macro-F1** | **0.844** | **0.441** |
| F1 neg | 0.854 | 0.226 |
| F1 neu | 0.751 | 0.824 |
| F1 pos | 0.927 | 0.273 |
| Support neg | 201 | 38 |
| Support neu | 232 | 146 |
| Support pos | 567 | 10 |

### 5.4 Matrices de confusion — domaine réel

**MARBERTv2** (macro-F1 = 0.441)
```
               pred_neg  pred_neu  pred_pos
true_neg  (38)     6        32         0      ← recall 0.158
true_neu (146)     8       129         9
true_pos  (10)     1         6         3      ← recall 0.300
```

**DarijaBERT** (macro-F1 = 0.439)
```
               pred_neg  pred_neu  pred_pos
true_neg  (38)    17        21         0      ← recall 0.447 (meilleur)
true_neu (146)    29       105        12
true_pos  (10)     2         6         2
```

**QARIB** (macro-F1 = 0.421)
```
               pred_neg  pred_neu  pred_pos
true_neg  (38)     8        30         0
true_neu (146)    15       117        14
true_pos  (10)     2         5         3
```

**xlm-t** (macro-F1 = 0.413)
```
               pred_neg  pred_neu  pred_pos
true_neg  (38)    24        12         2      ← recall 0.632 (le plus élevé)
true_neu (146)    52        72        22      ← sous-prédit neu
true_pos  (10)     4         2         4
```

**camelbert-da** (macro-F1 = 0.306)
```
               pred_neg  pred_neu  pred_pos
true_neg  (38)    35         1         2      ← recall 0.921 mais P=0.241
true_neu (146)   103        29        14      ← confusion massive neg/neu
true_pos  (10)     7         0         3
```

**Pattern commun :** tous les modèles peinent sur la classe neutre-vs-négative en contexte broadcast. Les stratégies divergent — xlm-t et camelbert-da sur-prédisent neg, les encoders fine-tunés sur-prédisent neu — mais aucun ne trouve l'équilibre.

### 5.4b Matrices de confusion — HACA annotations v2 (neg=63, neu=111, pos=20)

**MARBERTv2** (macro-F1 = 0.380)
```
               pred_neg  pred_neu  pred_pos
true_neg  (63)    10        53         0      ← recall 0.159
true_neu (111)     5        97         9
true_pos  (20)     1        16         3      ← recall 0.150
```

**DarijaBERT** (macro-F1 = 0.415)
```
               pred_neg  pred_neu  pred_pos
true_neg  (63)    26        37         0      ← recall 0.413
true_neu (111)    19        80        12
true_pos  (20)     3        15         2
```

**QARIB** (macro-F1 = 0.386)
```
               pred_neg  pred_neu  pred_pos
true_neg  (63)    14        48         1      ← recall 0.222
true_neu (111)     9        89        13
true_pos  (20)     2        15         3
```

**xlm-t** (macro-F1 = 0.489)
```
               pred_neg  pred_neu  pred_pos
true_neg  (63)    39        18         6      ← recall 0.619
true_neu (111)    34        62        15
true_pos  (20)     8         5         7      ← recall 0.350 (best pos recall)
```

**camelbert-da** (macro-F1 = 0.374)
```
               pred_neg  pred_neu  pred_pos
true_neg  (63)    57         3         3      ← recall 0.905 but P=0.396
true_neu (111)    73        26        12      ← recall 0.234 catastrophique
true_pos  (20)    14         2         4
```

### 5.5 Analyse de l'écart

**1. Registre :** Les tweets MAC utilisent une darija explicite et émotionnelle ("هذا سيء جداً", "واو رائع"). Le contenu broadcast utilise un registre mesuré et journalistique ("فيه مشاكل كبيرة", "لم تستفد الطبقة الوسطى"). Les modèles fine-tunés sur MAC n'ont jamais vu cette forme indirecte de critique.

**2. Déséquilibre de classes :** Le jeu domaine réel est 75% neutre. Le jeu MAC a une majorité positive (56,7%). Les frontières de décision apprises en entraînement sont biaisées vers pos/neg — recall neg moyen des fine-tunés = 0.27 vs recall neg de xlm-t = 0.63 (xlm-t n'a pas appris ce biais).

**3. Bruits ASR :** ~36 des 194 utterances viennent de fichiers SRT avec une transcription automatique médiocre (fichiers 1.srt, 2.srt, 6.srt). Ces fragments ininterprétables gonflent artificiellement la classe neutre et bruitent les prédictions.

**4. Effet corpus :** MAC est composé de tweets politiques et sociaux. Le contenu broadcast inclut des émissions religieuses, des documentaires économiques, des débats parlementaires — domaines absents des données d'entraînement.

**5. Pourquoi xlm-t généralise mieux :** xlm-t est un modèle multilingue non fine-tuné sur un corpus Arabic spécifique — il n'a pas intériorisé les corrélations shortcut du jeu MAC (mots d'argot positifs, emojis). En contrepartie, sa performance absolue reste inférieure à MARBERTv2 sur les tweets.

### 5.6 Implications pour la HACA

L'écart de −0.30 à −0.41 est **structurel**, pas accidentel. Il touche tous les modèles, y compris les meilleurs. Les options concrètes :

1. **Court terme — adaptation fine-tuning :** Annoter 500–1 000 utterances broadcast supplémentaires et affiner MARBERTv2 sur un corpus mixte (MAC tweets + broadcast). La séparation des distributions train/test suggère que 300–500 exemples broadcast par classe suffiraient à combler la majorité de l'écart (objectif : macro-F1 ≥ 0.70 sur le jeu domaine réel).
2. **Moyen terme — jeu de test plus robuste :** Créer un jeu domaine réel de 500+ utterances avec plusieurs annotateurs et kappa de Cohen. L'actuel (194 utt., 1 annotateur) doit être considéré comme une étude pilote.
3. **Remarque sur la classe pos :** Avec seulement 10 exemples positifs, la F1 pos est haute-variance. Le sentiment positif est structurellement rare dans le contenu d'information broadcast. Une stratégie d'annotation ciblée est nécessaire.
4. **Décision de déploiement :** Si le fine-tuning broadcast n'est pas réalisable immédiatement, xlm-t offre le meilleur compromis généralisation / couverture multilingue (gap −0.310 vs −0.403 pour MARBERTv2), bien qu'à un niveau absolu inférieur.

---

## 6. Recommandations de déploiement

### Option A — Routage max-précision (recommandé)

Un modèle différent par langue, choisir le meilleur F1.

| Langue détectée | Modèle | Macro-F1 public | Latence |
|---|---|---|---|
| darija\_ar | MARBERTv2 fine-tuné | 0.844 | 2.5 ms |
| arabizi | DarijaBERT-arabizi fine-tuné | 0.983 | 3.9 ms |
| francais | distilcamembert | 0.949 | 420.9 ms |
| msa | camelbert-da | 0.924 | 163.7 ms |

**Mémoire totale :** ~2 Go de poids (fp16). Applicable sur tout serveur avec GPU 8 Go.
**Limitation :** 4 modèles à maintenir. La F1 domaine réel de MARBERTv2 tombe à 0.441 — adaptation broadcast nécessaire.

### Option B — Modèle unique (min-coût)

xlm-t couvre toutes les langues : F1 moyen = 0.767. Perd ~7–21 points selon la langue. À réserver si les ressources d'inférence sont très contraintes.

### Option C — Encodeur de secours LLM

Utiliser Atlas-Chat-2B pour darija\_ar et arabizi si aucun fine-tuning n'est possible :
- darija_ar : 0.727 (−11.7 pts vs MARBERTv2)
- arabizi : 0.978 (−0.5 pts vs DarijaBERT-arabizi, mais 47× plus lent)

L'écart sur la darija est trop large pour recommander cette option en production, sauf si le fine-tuning est impossible.

---

## 7. Prochaines étapes

1. **Adaptation broadcast :** Annoter ~500 utterances broadcast supplémentaires et relancer le fine-tuning de MARBERTv2 avec ce corpus mixte pour combler l'écart domaine réel (objectif : macro-F1 ≥ 0.70 sur le jeu domaine réel).
2. **Couverture française :** Le jeu domaine réel ne contient aucune utterance française — les émissions HACA en français ne sont pas représentées. Si la HACA régule des contenus francophones, construire un sous-jeu domaine réel français.
3. **Test arabizi broadcast :** Les SRT actuels ne contiennent pas d'arabizi. Si des émissions utilisent l'arabizi (rare à la TV), le tester séparément.
4. **Plots et grille pondérée :** Regénérer la heatmap macro-F1 (modèle × langue), le scatter coût-précision, le radar des finalistes, et la grille pondérée (Précision 50% + Intégration 20% + Coût 20% + Couverture 10%) avec tous les modèles incluant Atlas-Chat 2B et 9B.

---

## 8. Adaptation au domaine — HACA-Sent v3 (juin 2026)

Suite à la §5, le jeu d'entraînement in-domaine a été **entièrement reconstruit** pour combler
l'écart broadcast, en visant surtout la classe positive (F1 = 0.19, le vrai goulot).

### 8.1 Ce qui a changé
- **Rubrique v3 canonique** (content-valence) : on étiquette ce que le contenu *décrit*, pas le
  ton du présentateur ([ANNOTATION_RUBRIC_V3.md](ANNOTATION_RUBRIC_V3.md)).
- **Ré-extraction qualité** (`src/build_haca_pool.py`) : découpage des longs paragraphes
  YouTube, filtre de bruit ASR, et filtre d'anti-fuite (5-grammes) contre le jeu de test
  → **1199 utterances propres**.
- **Étiquetage manuel** des 1199 (neu 973 / neg 191 / pos 35) + **189 synthétiques**
  pos-pondérées (129 pos) → `haca_train_v3.csv` : **1388 lignes** (neu 993 / neg 231 / pos 164).
- Fine-tuning avec **perte focale pondérée par classe + sur-échantillonnage ×3–4 de pos**.
- Provenance et limites détaillées : [HACA_DATASET.md](HACA_DATASET.md).

### 8.2 Résultats — jeu domaine réel v2 (neg 63 / neu 111 / pos 20)

| Modèle | macro-F1 | (calibré) | F1 neg | F1 neu | **F1 pos** | rappel pos | darija\_ar |
|---|---|---|---|---|---|---|---|
| MARBERTv2-haca (MAC + broadcast) | 0.459 | 0.521 | 0.500 | 0.744 | 0.133 | 0.10 | 0.835 |
| MARBERTv2-haca-only (broadcast seul) | 0.469 | ~0.52 | 0.522 | 0.755 | 0.129 | 0.10 | 0.830 |
| DarijaBERT-haca | 0.453 | 0.523 | 0.465 | 0.708 | 0.188 | 0.15 | 0.767 |
| Atlas-Chat-2B (rubrique en prompt) | 0.493 | — | 0.619 | 0.537 | **0.322** | **0.70** | — |
| Atlas-Chat-9B (rubrique en prompt) | **0.504** | — | 0.584 | 0.634 | 0.292 | 0.35 | — |
| *(rappel §5) MARBERTv2 brut* | 0.380 | 0.503 | 0.226 | 0.824 | 0.273 | 0.30 | 0.844 |

### 8.3 Constatations
1. **Aucune régression sur darija_ar** (0.83) : les modèles restent excellents sur le texte propre.
2. **Tous plafonnent à ~0.50–0.52** sur le broadcast. **L'objectif 0.70 n'est pas atteint.**
3. **La classe pos reste le goulot pour les encoders** (F1 0.13–0.19) malgré pos ×4.7, perte
   focale et sur-échantillonnage. Retirer MAC (haca-only) n'a rien changé — l'hypothèse selon
   laquelle MAC polluait le signal positif était fausse.
4. **Le LLM avec rubrique gère bien mieux la classe positive** (F1 0.29–0.32, rappel jusqu'à
   0.70) car il *applique* la consigne au lieu de devoir l'apprendre.
5. **Encoder calibré (~0.52) et LLM-rubrique (~0.50) à égalité en macro-F1**, mais profils
   opposés : l'encoder excelle sur neu/neg et est aveugle au positif ; le LLM est le seul à
   détecter pos.

### 8.4 Pourquoi le plafond — le jeu de test est le facteur limitant
L'inspection des 20 positifs du test l'explique :
- **sources partiellement absentes** : certains positifs viennent de fichiers (`f1`, `d1`, `14`,
  un `9.srt` « Sahara 2025 ») qui **ne sont pas dans le corpus SRT actuel** — le modèle ne les a
  jamais vus ;
- **frontière pos/neu différente de la rubrique v3** : le test compte comme « pos » des énoncés
  que la rubrique v3 classe neu (un service qui existe, l'interdiction de fumer dans le train,
  une incitation à l'investissement) ;
- **20 positifs, un seul annotateur** → F1 pos à très haute variance.

Le plafond est donc surtout un problème **d'alignement d'annotation + sources non vues + petit
échantillon**, pas un problème de modèle. Viser 0.70 sur ce jeu précis n'est pas équitable tant
qu'il n'est pas reconstruit sous la même rubrique.

#### 8.4 bis — Étude de cohérence d'annotation (v2 humain vs v3 rubrique)
Pour quantifier la subjectivité, les **194 mêmes énoncés** ont été ré-étiquetés sous la
rubrique v3 (`src/apply_annotations_domaine_reel_v3.py` → `domaine_reel_v3.csv`).

| | neg | neu | pos |
|---|---|---|---|
| v2 (humain) | 63 | 111 | 20 |
| v3 (rubrique, Claude) | 45 | 133 | 16 |

- **Accord 88.7 %, kappa de Cohen = 0.784** (« substantiel »).
- Les 22 changements vont **tous vers neu** (18 neg→neu, 4 pos→neu) : fragments ASR illisibles
  et cas-limites (un service qui existe, une interdiction de fumer, un mécanisme de préférence).
- **La classe positive passe de 20 à 16** — soit 20 % de variation sur la classe critique.

Conséquence : on **ne peut pas mesurer de façon fiable une classe positive de 16–20 exemples**
quand une annotation à kappa 0.78 la fait bouger de 20 %. C'est la limite dure du benchmark
actuel. `domaine_reel_v3.csv` est un **artefact de cohérence** (annoté par Claude, même rubrique
que l'entraînement → biais d'alignement) à faire **adjuger par un second annotateur humain**
avant de servir de gold officiel ; le v2 humain reste le gold indépendant.

**Ré-évaluation sur le gold v3 — une hypothèse réfutée.** On a re-scoré les modèles sur v3 :

| Modèle | macro-F1 (v2) | macro-F1 (v3) | F1 pos (v3, n=16) |
|---|---|---|---|
| MARBERTv2-haca — défaut | 0.459 | **0.452** | 0.077 |
| MARBERTv2-haca — calibré | 0.521 | **0.518** | 0.138 |
| Atlas-Chat-2B — rubrique | 0.493 | **0.452** | 0.289 |

Contrairement à ce qu'on attendait, **ré-étiqueter sous une rubrique cohérente n'améliore pas
l'encoder** (0.452 vs 0.459). Le plafond de l'encoder est donc **invariant à la version
d'annotation** : l'écart n'est **pas** principalement un problème d'alignement (l'intuition de
§8.4 est en partie réfutée), mais **structurel** — tâche difficile pour un encoder compact +
classe positive non mesurable (1–2 corrects sur 16–20 quelle que soit la rubrique). Le LLM, lui,
baisse sur v3 (0.452 vs 0.493) car v3 est plus neu-lourd et il **sous-prédit neu** (rappel
neu = 0.41) ; son avantage sur v2 venait en partie de la moindre proportion de neutre. Sur v3
(proportion de neutre proche du réel), **encoder calibré ≈ LLM-rubrique (~0.45–0.52)**.

**Conclusion affinée :** sur ce contenu broadcast, tous les modèles plafonnent à ~0.45–0.52
**quelle que soit la rubrique** ; la classe positive (16–20 ex., F1 0.08–0.32) est non mesurable.
Ce plafond n'est réparable ni par ré-étiquetage ni par plus de données d'entraînement : il faut
un **jeu d'évaluation plus grand, équilibré et multi-annotateur**, ce qui suppose d'élargir le
corpus source (aujourd'hui limité à 12 fichiers pauvres en positif).

#### 8.4 ter — Étude de cas « facile ≠ bon » (jeu équilibré v4)
Pour atteindre les effectifs du protocole sur les 12 fichiers actuels, on a construit
`domaine_reel_v4_balanced` (450 énoncés, **neg 150 / neu 200 / pos 100**) en complétant les
194 énoncés réels par **256 énoncés synthétiques** (voir HACA_TEST_V4_DATASHEET.md). Résultat
**contre-intuitif** : tout le monde « réussit ».

| Modèle | macro-F1 (v2 = sous-ensemble réel) | macro-F1 (v4 complet) | F1 pos (v2 → v4) |
|---|---|---|---|
| MARBERTv2-haca (encoder) | 0.459 | **0.796** | 0.133 → **0.770** |
| Atlas-Chat-2B (LLM, zero-shot) | 0.493 | **0.714** | 0.322 → **0.667** |

**Pourquoi le saut — et pourquoi il ne valide PAS le jeu.** Le sous-ensemble `human_gold` de v4
**est exactement** `domaine_reel_v2` ; sur lui les scores restent ~0.46–0.49. Le saut vient
**uniquement** des 256 énoncés synthétiques, qui sont **propres, non ambigus et prototypiques**
(« المنتخب الوطني تأهل لكأس العالم », « أسعار الخضر طلعات »…) — **faciles pour n'importe quel
modèle**. Le broadcast réel est l'inverse : ASR bruité, valence subtile, majorité neutre.

Deux effets distincts, isolables :
1. **Facilité** (profite à *tous*, y compris au LLM **zero-shot** qui n'a jamais vu ce
   synthétique) — d'où le saut d'Atlas-Chat-2B (+0.34 sur pos) ;
2. **Mémorisation de style** (bonus *en plus* pour les encoders, entraînés sur ce synthétique)
   — l'encoder gagne **+0.64** sur pos, soit ~2× le LLM. Cet écart EST la mémorisation ; le saut
   commun est la facilité.

**Leçon :** un jeu peut être **plus grand, équilibré, et faire monter tous les scores à 0.7+**
tout en étant un **moins bon benchmark** que les 194 exemples réels. La **taille et l'équilibre
ne suffisent pas** ; un bon jeu doit **discriminer** et refléter la difficulté réelle. Le chiffre
honnête reste ~0.46–0.49 (`domaine_reel_v2`). `domaine_reel_v4_balanced` est donc un **diagnostic
par classe**, **pas** un gold — l'ingrédient manquant n'est pas le *nombre* d'exemples, mais des
exemples **réels, difficiles, représentatifs et annotés par des humains**.

### 8.5 Recommandation révisée
- **Déploiement** : pour une tâche de *content-valence*, privilégier un **LLM instructable avec
  la rubrique en prompt** (Atlas-Chat) si la détection du positif compte — c'est le seul à la
  gérer (rappel pos 0.35–0.70), au prix de la latence (~185–450 ms vs ~3 ms encoder). Sinon, un
  **encoder fine-tuné + calibration de seuils** offre le même macro-F1 (~0.52) à 100× la vitesse,
  mais avec un angle mort sur le positif.
- **Étape réelle suivante** : **reconstruire le jeu de test broadcast** sous la rubrique v3 —
  mêmes fichiers sources que l'entraînement, ≥ 50 positifs, deux annotateurs avec kappa de Cohen.
  Le jeu actuel (20 pos, pilote, mono-annotateur) ne mesure pas fiablement la classe positive.

### 8.6 Conclusion
Sur le contenu broadcast, **tous les modèles plafonnent à ~0.45–0.52** — encoder calibré et
LLM-rubrique inclus — et ce **quelle que soit la rubrique d'annotation** (vérifié sur v2 et v3).
L'encoder reste aveugle au positif ; le LLM le détecte mieux (rappel pos 0.70–0.75) mais
sous-prédit le neutre, si bien que son avantage s'évapore dès que le jeu est neu-lourd (le cas
réel). La *content-valence* est une tâche de **suivi de consigne** qui favorise un LLM
instructable **quand le positif compte**, mais pour un flux majoritairement neutre l'**encoder
calibré (~0.52, 100× plus rapide)** est le choix pragmatique. Surtout, le levier décisif n'est
**ni le modèle, ni la quantité de données d'entraînement, ni le ré-étiquetage** : c'est la
**taille et la représentativité du jeu d'évaluation** — une classe positive de 16–20 exemples
(kappa 0.78) ne peut tout simplement pas être mesurée. Élargir le corpus source au-delà des 12
fichiers actuels (pauvres en positif) est le préalable à tout progrès chiffrable.

---

## Annexes

- Résultats détaillés par classe : `results/*.json`
- Résumé agrégé : `results/summary.csv`
- Jeu domaine réel (annotations strictes) : `data/test_sets/domaine_reel.csv`
- Jeu Gemini (annotations larges, 65/97/32) : `data/test_sets/gemini.csv`
- Jeu HACA v2 (annotations HACA, 63/111/20) : `data/test_sets/domaine_reel_v2.csv` — SHA-256 : `32b8e75d920858686adbfc2e8126c9ce72375a480d4ba88f92fc0b55938e098d`
- Script d'annotation v2 : `src/apply_annotations_gemini.py`
- Règles d'annotation : [DOMAINE_REEL_ANNOTATION.md](DOMAINE_REEL_ANNOTATION.md)
- Pipeline SRT : [SRT_PIPELINE.md](SRT_PIPELINE.md)
- Analyse détaillée modèle par modèle : [RESULTS_ANALYSIS.md](RESULTS_ANALYSIS.md)
- Notes fine-tuning et LLM : [FINETUNING.md](FINETUNING.md)
