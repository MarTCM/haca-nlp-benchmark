# Option 1 — Calibration des seuils de décision

**Statut :** Implémenté et validé
**Script :** `src/calibrate_thresholds.py`
**Résultats :** `results/thresholds_marbertv2.json`
**Gain mesuré :** macro-F1 domaine réel : 0.441 → 0.503 (+0.062 sur le jeu complet, +0.047 en cross-validation)

---

## 1. Le problème que cette option résout

Après l'évaluation sur le jeu domaine réel (194 utterances broadcast), MARBERTv2 obtient un macro-F1 de **0.441** avec une **recall neg catastrophiquement basse (0.158)** : le modèle ne détecte que 15,8 % des vrais négatifs.

En inspectant les scores bruts, on constate que le modèle n'est pas aveugle aux négatifs — il leur attribue souvent un score neg de 0.15 à 0.40. Le problème est qu'il ne passe jamais à l'action, parce que le **seuil de décision par défaut est 0.50** pour toutes les classes. Une critique broadcast du type *"لم تستفد الطبقة الوسطى"* (la classe moyenne n'en a pas bénéficié) reçoit peut-être neg=0.30, neu=0.55, pos=0.15 — le modèle prédit neu, alors que neg=0.30 est déjà un signal clair.

La calibration ne modifie pas les poids du modèle. Elle ajuste uniquement la règle de décision appliquée aux scores existants.

---

## 2. Comment fonctionne la calibration — l'argmax décalé

### La règle par défaut (argmax standard)

Pour chaque utterance, le modèle produit trois scores softmax :

```
neg = 0.30,  neu = 0.55,  pos = 0.15
```

La prédiction par défaut est simplement la classe avec le score le plus élevé :

```
argmax({neg: 0.30, neu: 0.55, pos: 0.15}) → neu
```

### La règle calibrée (argmax décalé)

On soustrait un seuil propre à chaque classe avant de prendre l'argmax :

```
neg_adj = 0.30 - T_neg
neu_adj = 0.55 - 0.50   (T_neu est fixé à 0.50, pas de décalage)
pos_adj = 0.15 - T_pos
```

Si `T_neg = 0.15` :

```
neg_adj = 0.30 - 0.15 = +0.15
neu_adj = 0.55 - 0.50 = +0.05
pos_adj = 0.15 - 0.50 = -0.35
```

```
argmax({neg: +0.15, neu: +0.05, pos: -0.35}) → neg  ✓
```

En abaissant `T_neg` à 0.15, on dit au modèle : *"un score neg de 0.15 suffit à battre un score neu de 0.50"*. Le modèle déclenche la prédiction neg beaucoup plus tôt.

### Pourquoi cette formule est valide

L'argmax décalé est mathématiquement équivalent à déplacer les frontières de décision dans l'espace des probabilités. C'est une technique standard en apprentissage automatique connue sous le nom de **threshold-moving** ou **cost-sensitive prediction**. Elle ne change pas la distribution apprise — elle change seulement la façon dont on interprète cette distribution pour une nouvelle tâche ou un nouveau domaine.

---

## 3. Comment les seuils optimaux sont trouvés

On cherche les valeurs de `T_neg` et `T_pos` qui maximisent le macro-F1 sur les données disponibles. On fait une **recherche par grille** : on teste toutes les combinaisons de T_neg et T_pos dans {0.05, 0.10, 0.15, …, 0.60} — soit 144 combinaisons au total.

**Problème :** avec seulement 194 utterances, si on optimise les seuils sur les mêmes données qu'on évalue, on obtient un score gonflé (surapprentissage du jeu de test). Ce serait tricher.

**Solution : validation croisée à 5 plis (5-fold cross-validation)**

```
Données : 194 utterances, stratifiées (même proportion neg/neu/pos dans chaque pli)

Pli 1 :  [train : 155 utt] → grid search → T_neg*, T_pos*
          [test  :  39 utt] → évaluation avec T_neg*, T_pos* → F1_fold1

Pli 2 :  [train : 155 utt] → grid search → T_neg*, T_pos*
          [test  :  39 utt] → évaluation → F1_fold2

...  (5 fois)

Résultat honnête = moyenne(F1_fold1, ..., F1_fold5)
```

Chaque pli optimise sur des données différentes de celles sur lesquelles il évalue. La moyenne des 5 plis est une estimation non biaisée du gain réel en déploiement.

---

## 4. Résultats

### Thresholds optimaux trouvés

| Classe | Seuil par défaut | Seuil calibré |
|---|---|---|
| neg | 0.50 | **0.15** |
| neu | 0.50 | 0.50 (inchangé) |
| pos | 0.50 | 0.50 (inchangé) |

Le seuil pos n'a pas changé : le modèle n'a pas de signal pos sous-exploité sur ce corpus (il n'y a que 10 vrais positifs, classe trop petite pour guider l'optimisation de manière fiable).

### Macro-F1 avant / après

| Mesure | Valeur |
|---|---|
| Default macro-F1 (jeu complet) | 0.441 |
| Calibrated macro-F1 (jeu complet) | **0.503** |
| Gain (jeu complet) | +0.062 |
| **CV gain (estimation honnête)** | **+0.047** |
| CV std | ±0.094 → ±0.070 (la variance diminue aussi) |

### Détail par classe

| Classe | Métrique | Default | Calibré | Δ |
|---|---|---|---|---|
| neg | Précision | 0.400 | **0.545** | +0.145 |
| neg | Rappel | 0.158 | **0.316** | +0.158 |
| neg | F1 | 0.226 | **0.400** | +0.174 |
| neu | Précision | 0.772 | 0.800 | +0.028 |
| neu | Rappel | 0.884 | 0.877 | −0.007 |
| neu | F1 | 0.824 | 0.837 | +0.013 |
| pos | F1 | 0.273 | 0.273 | 0 |

Le rappel neg **double** (0.158 → 0.316). La précision neg augmente aussi (+0.145) parce que les nouvelles prédictions neg sont de vraies critiques, pas du bruit. La classe neu ne se dégrade presque pas.

---

## 5. Utilisation à l'inférence

Pour utiliser les seuils calibrés dans n'importe quel script d'inférence, remplacer le bloc de décision standard par :

```python
import json

# Charger les seuils une seule fois au démarrage
thresholds = json.load(open("results/thresholds_marbertv2.json"))["thresholds"]
# → {"neg": 0.15, "neu": 0.50, "pos": 0.50}

def predict_calibrated(raw_scores: dict, thresholds: dict) -> str:
    """
    raw_scores : {"neg": 0.30, "neu": 0.55, "pos": 0.15}
    thresholds : {"neg": 0.15, "neu": 0.50, "pos": 0.50}
    """
    shifted = {cls: raw_scores[cls] - thresholds[cls] for cls in raw_scores}
    return max(shifted, key=shifted.get)
```

Le coût supplémentaire est exactement **0** — même modèle, mêmes poids, même vitesse d'inférence. C'est une opération arithmétique sur trois nombres.

---

## 6. Limites de cette approche

**Ce que la calibration corrige :**
- Le biais de distribution entre tweets (MAC, pos-dominant) et broadcast (neu-dominant)
- Les frontières de décision trop agressives sur la classe neu

**Ce que la calibration ne peut pas corriger :**
- Les cas où le modèle attribue un score neg proche de 0 à un vrai négatif (il ne peut pas récupérer ce qu'il ne voit pas du tout)
- Le bruit ASR : des utterances transcrites de façon incorrecte restent incorrectement classées
- La classe pos : trop peu d'exemples (10) pour que l'optimisation soit fiable. T_pos reste à 0.50 faute de signal.
- Le plafond de cette approche : on estime que la calibration seule ne peut pas dépasser ~0.55 de macro-F1 sur ce corpus. Pour atteindre ≥ 0.70, il faut de l'adaptation au domaine par fine-tuning (Option 2).

**Sur la validité des seuils sauvegardés :**
Les seuils dans `results/thresholds_marbertv2.json` sont entraînés sur le jeu complet (194 utterances). Ils sont optimaux pour ce jeu spécifique et légèrement suroptimisés par rapport à leur performance réelle en déploiement — l'estimation honnête est le CV gain (+0.047), pas le gain full-dataset (+0.062).

---

## 7. Résumé en une phrase

En abaissant le seuil de décision neg de 0.50 à 0.15, le modèle détecte deux fois plus de vrais négatifs dans le contenu broadcast, pour un gain de +4.7 à +6.2 points de macro-F1, sans aucune modification des poids du modèle.
