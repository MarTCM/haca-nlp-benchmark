# Guide complet du projet — Analyse de tonalité broadcast pour la HACA

> Document pédagogique : explique **tout le projet** de bout en bout, le *pourquoi* de chaque
> étape, les concepts en langage simple, et une section **Questions/Réponses** pour la
> soutenance. (Disponible en français ; demander une version anglaise si besoin.)

---

## 0. Résumé en une page (pour la soutenance)

**Le problème.** La HACA veut analyser automatiquement la **tonalité** (négatif / neutre /
positif) du contenu diffusé à la radio/télé en darija. Des modèles existants marchent très bien
sur des **tweets** (macro-F1 ≈ 0.84) mais s'**effondrent** sur le **vrai contenu broadcast**
(≈ 0.46–0.52).

**Ce qu'on a fait.** On a reconstruit proprement le jeu de données in-domaine, ré-entraîné les
modèles avec les bonnes techniques, comparé encoders et LLM, puis **diagnostiqué** pourquoi ça
plafonne — et on a **prouvé** la cause par une expérience (ré-annotation + ré-évaluation).

**Ce qu'on a trouvé.** Le plafond est **structurel**, pas un défaut de modèle : la tâche est
subtile, et surtout la **classe positive du jeu de test est trop petite (16–20 exemples) pour
être mesurée**. Plus de données d'entraînement ou un ré-étiquetage **ne changent rien**.

**La solution livrée.** Au lieu d'un classifieur fragile par phrase, on a construit un **système
de tonalité au niveau émission/segment**, qui (1) **filtre** l'ASR illisible au lieu de
l'étiqueter, (2) **agrège** les phrases en segments pour noyer le bruit, et (3) **signale pour
révision humaine** les cas incertains. C'est **fiable** et c'est ce qu'un régulateur utilise.

**La phrase clé.** *Le levier décisif n'est ni le modèle ni la quantité de données, mais la
qualité du jeu d'évaluation et la bonne conception du système.*

---

## 1. Le contexte et le problème

La **HACA** (Haute Autorité de la Communication Audiovisuelle) régule les médias au Maroc. On
veut un outil qui lit la transcription d'une émission et dit si le **contenu** est négatif,
neutre ou positif — pour suivre la tonalité des programmes.

Les transcriptions arrivent sous forme de fichiers **SRT** (sous-titres) issus d'une
reconnaissance vocale automatique (**ASR**) souvent **bruitée** (du texte mal transcrit,
parfois illisible). Le contenu est de la **darija marocaine en écriture arabe**.

**Le problème central observé au départ :** un modèle (MARBERTv2) affiné sur des tweets
marocains obtient **0.84** sur des tweets, mais seulement **~0.38** sur le vrai broadcast — une
chute énorme. C'est ce qu'on appelle un **« domain gap »** (écart de domaine).

---

## 2. Les concepts clés, en langage simple

- **Tonalité « content-valence ».** On n'étiquette **pas l'émotion du présentateur**, mais ce
  que le **contenu décrit**. Une émission qui rapporte calmement un scandale de corruption est
  **négative** (le contenu décrit quelque chose de mauvais), même si l'animateur reste neutre.
  C'est ça que la HACA régule : le contenu diffusé.
- **Domain gap.** Un modèle entraîné sur un type de texte (tweets émotionnels) échoue sur un
  autre (langage journalistique mesuré). Les indices ne sont pas les mêmes.
- **Déséquilibre de classes.** Dans le broadcast d'info, le **neutre** domine, le **négatif**
  est fréquent, le **positif** est **rare**. Un modèle a du mal à apprendre une classe qu'il
  voit peu.
- **Macro-F1.** La métrique principale : la moyenne des scores F1 des 3 classes, **à poids
  égal**. Donc si une classe (le positif) est mauvaise, elle tire toute la note vers le bas —
  même si elle est rare. C'est volontaire : on ne veut pas qu'un modèle qui dit « neutre »
  partout ait une bonne note.
- **Fine-tuning (affinage).** On part d'un modèle de langue pré-entraîné et on l'entraîne un
  peu plus sur nos données étiquetées pour la tâche précise.
- **Perte focale + sur-échantillonnage.** Deux techniques pour forcer le modèle à faire
  attention aux **classes rares** (ici le positif) pendant l'entraînement.
- **Calibration de seuils.** Au lieu de prendre toujours la classe la plus probable, on baisse
  le seuil de décision d'une classe sous-détectée (ex. négatif) pour qu'il la prédise plus
  souvent. Gratuit, +0.05 environ.
- **Kappa de Cohen.** Mesure l'**accord entre deux annotateurs** (0 = hasard, 1 = parfait).
  Sert à savoir si la tâche d'étiquetage est fiable ou subjective.

---

## 3. La démarche, étape par étape (l'histoire du projet)

### 3.1 Diagnostic de départ
On regarde *où* ça casse. Sur le vrai broadcast, le détail par classe montre : **neutre** ≈ 0.75
(ok), **négatif** ≈ 0.63 (correct), **positif** ≈ 0.19 (cassé). Le coupable principal est la
**classe positive**, et deux causes : (a) trop peu d'exemples positifs, (b) des fragments ASR
illisibles qui polluent les données.

### 3.2 Reconstruction des données (le gros du travail)
1. **Rubrique d'annotation v3** (`ANNOTATION_RUBRIC_V3.md`) : on fixe une règle claire et
   cohérente (content-valence) avec des conventions pour les cas-limites.
2. **Ré-extraction qualité** (`build_haca_pool.py` + `asr_quality.py`) : on découpe mieux les
   transcriptions, on **filtre** les fragments ASR illisibles, et on retire tout ce qui
   **chevauche le jeu de test** (anti-fuite par « 5-grammes ») → **1199 phrases propres**.
3. **Annotation manuelle** des 1199 phrases sous la rubrique v3 → **973 neutre / 191 négatif /
   35 positif** (le positif est vraiment rare dans ce corpus).
4. **Données synthétiques** (`synthetic_haca.py`) : on **génère** des phrases broadcast
   positives (et quelques neg/neu) pour donner au modèle de quoi apprendre la classe rare →
   **189 phrases** (dont 129 positives).
5. **Assemblage** (`build_haca_train.py`) → `haca_train_v3.csv` : **1388 phrases** (993 / 231 /
   164). Le positif passe de 35 à 164 (×4.7).

### 3.3 Entraînement (`finetune.py`)
On affine deux encoders (`marbertv2-haca`, `darijabert-haca`) avec **perte focale pondérée +
sur-échantillonnage ×3 du positif** pour combattre le déséquilibre. On ajoute aussi une variante
**« in-domaine seule »** (sans les tweets MAC) pour tester si les tweets polluaient le signal.

### 3.4 Évaluation + un LLM « avec la règle en consigne » (`eval_llm_rubric.py`)
On évalue tout sur le jeu de test **humain** de référence (`domaine_reel_v2`, 194 phrases). On
ajoute un **LLM darija (Atlas-Chat)** à qui on **donne la rubrique dans le prompt** — l'intuition
étant que la content-valence est une tâche de **suivi de consigne**, où un LLM instruit pourrait
battre un petit encoder.

### 3.5 Les résultats — et le plafond
| Modèle | broadcast (macro-F1) | F1 positif | darija propre |
|---|---|---|---|
| MARBERTv2-haca | 0.459 → 0.521 (calibré) | 0.13 | 0.835 |
| DarijaBERT-haca | 0.453 → 0.523 (calibré) | 0.19 | 0.767 |
| Atlas-Chat-2B (rubrique) | 0.493 | **0.32** | — |
| Atlas-Chat-9B (rubrique) | 0.504 | 0.29 | — |

Constats : **pas de régression** sur le texte propre (≈ 0.83) ; tout le monde **plafonne à
~0.45–0.52** ; l'encoder reste **aveugle au positif** ; seul le LLM le détecte un peu mieux.

### 3.6 On forme une hypothèse et on la teste (ré-annotation v3)
Hypothèse : « le plafond vient d'un mauvais alignement d'annotation ». Pour tester, on
**ré-étiquette les mêmes 194 phrases** sous la rubrique v3 (`domaine_reel_v3.csv`).
Résultat : **accord 88.7 %, kappa = 0.784**, mais le **positif bouge de 20 → 16**. Donc même
avec une règle écrite, l'annotation est subjective et la classe positive est instable.

### 3.7 On réfute l'hypothèse (ré-évaluation sur v3)
On ré-évalue les modèles sur ce jeu ré-aligné. **Surprise : l'encoder ne bouge pas**
(0.452 vs 0.459). Donc le plafond est **invariant à la rubrique** → ce **n'est pas** un problème
d'alignement, c'est **structurel**. (Bonne démarche scientifique : on a testé notre propre
explication et on l'a rejetée.)

### 3.8 Étude de cas « facile ≠ bon » (jeu équilibré v4)
On construit un jeu **équilibré** (150 neg / 200 neu / 100 pos) en complétant le réel par du
synthétique. **Tout le monde saute à 0.7+** (MARBERTv2 0.80, Atlas-Chat 0.71). Mais ça **ne
valide pas** le jeu : sa partie réelle reste à ~0.46 ; le saut vient des phrases synthétiques
**faciles** (prototypiques). **Un jeu plus grand et équilibré peut être un *moins bon*
benchmark** s'il est facile et non représentatif.

---

## 4. La conclusion scientifique

1. La **content-valence est une tâche de suivi de consigne** : elle favorise un LLM instruit
   quand le positif compte, mais un petit encoder reste compétitif et 100× plus rapide.
2. Le **plafond ~0.45–0.52 est structurel** : il ne bouge ni avec plus de données, ni avec un
   ré-étiquetage, ni avec une rubrique différente.
3. La **classe positive (16–20 exemples) est non mesurable** : un kappa de 0.78 la fait bouger
   de 20 %. On ne peut pas mesurer une classe aussi petite.
4. Le **vrai verrou** n'est pas le modèle ni la quantité de données, mais la **qualité du jeu
   d'évaluation** (taille, représentativité, double annotation humaine). Le protocole pour le
   construire est dans `TEST_SET_PROTOCOL.md`.

---

## 5. La solution déployable (le produit final)

Puisqu'un classifieur par phrase plafonne et qu'on a des SRT bruités, on a conçu le **bon
système** (`haca_pipeline.py`, détaillé dans `DEPLOYMENT.md`) :

```
SRT (bruité) → découpage → FILTRE QUALITE (on jette l'illisible, on ne l'étiquette pas)
            → encoder affiné + seuils calibrés
            → AGREGATION par segment et par émission
            → tonalité + confiance + couverture + drapeaux « à revoir »
```

Trois idées qui rendent ça **viable** malgré le plafond :
1. **Filtre qualité (abstention)** : on **n'étiquette pas le bruit ASR**, on l'exclut → meilleure
   précision.
2. **Agrégation au niveau segment/émission** : le bruit se **moyenne**, et c'est ce qu'un
   régulateur consomme (la tonalité d'une émission, pas d'un cue de 5 s).
3. **Confiance + révision humaine** : les cas incertains sont **signalés**, pas devinés.

Sorties : une **timeline** par émission, un **JSON** détaillé, et un **CSV tableau de bord**
(une ligne par segment) importable dans Excel/Power BI. Test validé : le documentaire sur la
corruption sort **négatif**, les fichiers très bruités sortent avec **couverture basse +
drapeau** (bon comportement : on ne devine pas sur du bruit).

### Extension multilingue + tableau de bord

Le système a été étendu pour traiter **trois langues** de SRT, pas seulement le darija en
caractères arabes :

- **Filtre qualité multilingue** (`asr_quality.is_clean`) : il compte désormais les mots-outils
  **arabes ET français** et mesure un ratio de « vrai texte » (script arabe + latin) au lieu d'un
  ratio purement arabe. Un SRT français ou code-switché n'est donc plus rejeté à tort
  (`no_intelligible_content`). Comportement **inchangé** pour un SRT 100 % arabe.
- **Registre de modèles par langue** (`haca_pipeline.MODEL_REGISTRY`) : un modèle de tonalité par
  langue (checkpoint local **ou** modèle du Hub mis en cache). `pick_model_for_lang` choisit
  automatiquement : `arabe → marbertv2-haca`, `arabizi → darijabert-arabizi`,
  `francais → xlm-sentiment`.
- **Tableau de bord** (`src/dashboard_app.py`, Streamlit) : on charge un SRT, on clique sur
  « Lancer l'analyse », il **détecte la langue**, choisit le modèle (mode « Auto ») ou laisse
  l'utilisateur choisir (dont **API Cloud** via clé, URL et modèle, par défaut `glm-5.2` sur
  Z.ai), et affiche verdict + timeline + CSV. Quand l'API Cloud est sélectionnée, une case à
  cocher permet d'**utiliser le même appel pour le sujet** (économie de tokens). Le bouton
  évite les appels API accidentels. Détaillé dans `PIPELINE.md` §4b.

**Le français — et la tentative de fine-tune.** Faute de modèle français entraîné sur du broadcast
HACA, le défaut est `cardiffnlp/twitter-xlm-roberta-base-sentiment` (vrai 3 classes neg/neu/pos).
On a tenté de faire mieux en affinant un encoder français sur des données **auto-annotées par
Claude** (sans API LLM) :
- jeu d'éval « gold » = 90 énoncés réels de `emission_francaise.srt`, étiquetés à la main
  (`build_francais_gold.py`) — gelé, jamais entraîné ;
- **1re tentative** (143 phrases synthétiques propres) : **échec** — les deux fine-tunes
  (`camembert-haca`, `xlm-r-haca`) passent **sous** l'off-the-shelf (0.34 / 0.40 vs **0.453**),
  car trop peu de données et trop propres face à l'ASR bruité réel ;
- **2e tentative** : `synthetic_haca_fr_large.py` génère des **milliers** d'exemples (templates ×
  banques de slots, toujours sans API) + **augmentation « bruit ASR »** (accents perdus, homophones,
  mots manquants, `[Musique]`) pour coller au registre réel. Un premier run a semblé gagner
  (`xlm-r-haca` 0.486), mais le ré-entraînement **méthodologiquement correct** (split de validation
  par templates disjoints) est redescendu à **0.448** — car la validation reste *synthétique*.

**Conclusion (réglée).** Sur le gold de 90 énoncés (un seul annotateur), **toutes** les options
françaises tiennent dans **0.43–0.49**, soit **dans le bruit** (~3 énoncés) : off-the-shelf,
fine-tunes, ensemble — indistinguables. On ne peut pas les classer de façon fiable. Le défaut
français est donc `xlm-sentiment` (off-the-shelf, le plus simple/reproductible, que rien ne bat de
façon fiable) ; les fine-tunes + l'ensemble restent sélectionnables. Détails : `FINETUNING.md` §6.
La leçon de fond (comme pour l'arabe) est confirmée : le vrai verrou est la **donnée/éval réelle**,
pas le modèle ni la quantité de synthétique — **on ne peut pas « régler » par-dessus ce manque**.

---

## 6. Carte des fichiers (où est quoi)

| Fichier | Rôle |
|---|---|
| `report/ANNOTATION_RUBRIC_V3.md` | La règle d'étiquetage (content-valence) |
| `src/build_haca_pool.py`, `src/asr_quality.py` | Extraction + filtre qualité ASR + anti-fuite |
| `src/apply_annotations_haca.py` | Les 1199 étiquettes manuelles |
| `src/synthetic_haca.py` | Données synthétiques d'entraînement |
| `src/build_haca_train.py` → `data/test_sets/haca_train_v3.csv` | Le jeu d'entraînement final |
| `src/finetune.py` | Entraînement (perte focale, sur-échantillonnage) |
| `src/calibrate_thresholds.py` | Calibration des seuils |
| `src/eval_llm_rubric.py` | LLM darija avec la rubrique en prompt |
| `src/apply_annotations_domaine_reel_v3.py` | Ré-annotation + kappa (étude de cohérence) |
| `src/build_domaine_reel_v4.py`, `src/synthetic_haca_test.py` | Jeu équilibré v4 (diagnostic) |
| **`src/haca_pipeline.py`** | **Le système déployable (tonalité par segment, multilingue)** |
| **`src/dashboard_app.py`** | **Tableau de bord web (Streamlit) : upload SRT → langue + verdict** |
| `src/synthetic_haca_fr.py`, `src/synthetic_haca_fr_large.py` | Données FR synthétiques (auto-annotées ; + bruit ASR) |
| `src/build_francais_gold.py` → `data/test_sets/francais_haca_gold.csv` | Gold FR réel (90 énoncés, à la main) |
| `src/eval_francais_gold.py` | Éval FR sur le gold |
| `report/FINDINGS.md` (§8) | Tous les résultats + l'analyse |
| `report/DEPLOYMENT.md`, `report/PIPELINE.md` | La solution déployable + le tableau de bord |
| `report/FINETUNING.md` (§6) | Fine-tune français (tentatives + résultats) |
| `report/TEST_SET_PROTOCOL.md` | Comment construire un bon jeu de test |
| `report/HACA_DATASET.md`, `HACA_TEST_V4_DATASHEET.md` | Fiches de provenance des données |
| `notebooks/kaggle_haca_full.ipynb`, `kaggle_finetune_francais.ipynb` | Notebooks Kaggle (arabe ; fine-tune FR) |

---

## 7. Comment l'exécuter (reproduire)

1. Pousser la branche sur GitHub : `git push -u origin feat/haca-sent-v3`.
2. Sur Kaggle (GPU T4), ouvrir `notebooks/kaggle_haca_full.ipynb`, mettre l'URL du dépôt, tout
   exécuter → entraîne l'encoder, calibre, évalue sur les 3 jeux, affiche un tableau.
3. Lancer le système de tonalité :
   `python src/haca_pipeline.py --srt-dir data/raw/srt --model marbertv2-haca --csv tonality.csv`

---

## 8. Questions/Réponses pour la soutenance

**« Pourquoi vous n'avez pas atteint l'objectif de 0.70 ? »**
Parce que ce n'est **pas atteignable sur ce jeu de test précis**, et on l'a **prouvé** : le
plafond ~0.5 est invariant au modèle et à la rubrique, et la classe positive (16–20 exemples)
est trop petite pour être mesurée (kappa 0.78 → ±20 %). L'objectif n'était pas réaliste tant que
le jeu de test n'est pas reconstruit. C'est un **résultat**, pas un échec.

**« Les données synthétiques, ce n'est pas tricher ? »**
Pour l'**entraînement**, non : c'est une technique standard pour une classe rare, et c'est
**déclaré** (fiche de données). Pour le **test**, oui ce serait trompeur — c'est pourquoi on a
montré dans l'étude « facile ≠ bon » que le synthétique gonfle les scores, et on **garde le gold
humain** comme référence.

**« Pourquoi un LLM ferait mieux qu'un encoder ? »**
La content-valence demande de **raisonner sur une règle** (« est-ce que le contenu décrit quelque
chose de bon/mauvais ? »). Un LLM à qui on **donne la règle** l'applique ; un petit encoder doit
l'**apprendre** à partir de peu d'exemples. C'est pour ça que le LLM détecte mieux le positif
(rappel 0.70). Mais sur un flux majoritairement neutre, l'encoder calibré fait jeu égal et est
bien plus rapide.

**« Quelle est la valeur du projet si le score est bas ? »**
Deux livrables : (1) un **diagnostic rigoureux** qui dit *pourquoi* c'est dur et *quoi faire*
(reconstruire le jeu de test — protocole fourni) ; (2) un **système déployable** qui contourne le
problème (tonalité par émission + filtre qualité + révision humaine) et qui est **ce qu'un
régulateur utilise vraiment**. La « limite » du benchmark **motive** la conception du produit.

**« Pourquoi au niveau segment et pas par phrase ? »**
Parce que la HACA régule des **émissions**, pas des phrases de 5 secondes, et parce qu'**agréger
noie le bruit ASR** : un segment de 2 minutes sur la corruption est clairement négatif même si
des cues isolés sont illisibles.

**« Et le filtre qualité, pourquoi jeter du texte ? »**
Parce que forcer une étiquette sur du **bruit illisible** crée des erreurs. Mieux vaut
**s'abstenir** (le signaler « à revoir ») et garder une **haute précision** sur ce qui est
intelligible. C'est le bon comportement pour un outil d'aide à la décision.

**« Comment on améliorerait encore ? »** (par ordre d'impact)
1. **Meilleure ASR** (le plus gros levier : le modèle échoue surtout sur ce qu'il ne peut pas
   lire). 2. **Reconstruire le jeu de test** (protocole : plus de sources positives, double
   annotation humaine, kappa). 3. **Escalade vers un LLM local** sur les segments incertains.
