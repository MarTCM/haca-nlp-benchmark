# Solution de déploiement — Pipeline de tonalité HACA

**Outil :** `src/haca_pipeline.py` — auto-hébergé, niveau **segment/émission**, robuste à l'ASR
bruité. Ce n'est **pas** un classifieur par énoncé : c'est un système de **tonalité** exploitable.

**Contraintes prises en compte :** on-premise (ou via API Cloud optionnelle), sortie au niveau
segment/émission, fichiers SRT fournis tels quels (ASR bruité non contrôlé).

---

## 1. Pourquoi cette conception (et pas un classifieur par énoncé)
FINDINGS §8 a montré que la classification **par énoncé** plafonne à ~0.46–0.52 macro-F1 sur ce
contenu — surtout à cause de (a) **fragments ASR illisibles** qu'on ne devrait pas étiqueter, et
(b) une classe positive minuscule. Une solution déployable ne combat pas ces limites, elle les
**contourne** :

1. **Filtre qualité (abstention).** Les énoncés illisibles (`src/asr_quality.py`) sont **exclus
   du score**, pas mal-étiquetés → précision plus haute sur ce qui est gardé.
2. **Agrégation au niveau segment/émission.** On regroupe les énoncés en segments (fenêtres) et
   on agrège : le bruit se moyenne, et **c'est ce qu'un régulateur consomme** (la tonalité d'une
   émission, pas d'un cue de 5 secondes).
3. **Confiance + file de révision humaine.** Faible couverture / pas de majorité claire / faible
   confiance → **drapeau « à revoir »** au lieu d'une décision automatique.

Résultat : un système **fiable** (haute précision sur les segments confiants) et **utile**
(tonalité par émission + tableau de bord), au lieu d'un autoclassifieur fragile à 0.46.

## 2. Architecture
```
SRT (bruité) → segmentation → FILTRE QUALITE (exclut le garbled)
            → soit encoder fine-tuné + seuils calibrés (par énoncé propre)
            → soit API Cloud (LLM via Chat Completions, batch unique)
            → AGREGATION par fenêtre (segment) et sur tout le fichier (émission)
            → rapport : tonalité dominante + distribution + confiance + couverture + drapeaux
```
Réutilise : `srt_utils` (segmentation + détection de langue), `build_haca_pool` (segmentation),
`asr_quality` (filtre **multilingue** : densité de mots-outils arabes *et* français + ratio de
script, donc un SRT français/code-switché n'est plus rejeté à tort), `calibrate_thresholds`
(seuils).

**Modèles (registre `MODEL_REGISTRY` dans `haca_pipeline.py`)** : un modèle par langue, choisi
automatiquement (`detect_lang`) en mode **auto** du tableau de bord —
`arabe → marbertv2-haca` (conseillé ; ou `darijabert-haca` licence permissive, `qarib`,
`marbertv2`), `arabizi → darijabert-arabizi`, `francais → xlm-sentiment`
(`cardiffnlp/twitter-xlm-roberta-base-sentiment`, vrai 3 classes neg/neu/pos ; alternative
`distilcamembert`, modèle 5★ d'avis qui écrase la classe neutre). Les modèles français sont des
modèles du Hub mis en cache localement, **non** fine-tunés HACA → verdicts français plus faibles,
sans seuils calibrés. Seuils calibrés via `results/thresholds_<model>.json` quand disponibles.

**Alternative API Cloud** : un LLM via endpoint compatible OpenAI Chat Completions (Z.ai, OpenRouter,
OpenAI, Groq…) peut remplacer les encodeurs par langue. Par défaut : `glm-5.2` sur
`https://api.z.ai/api/paas/v4/chat/completions`. Les énoncés sont envoyés en un seul batch
(économie de tokens) ; la détection du sujet peut être combinée au même appel pour économiser
davantage. Un bouton manuel **« Lancer l'analyse »** évite les appels accidentels.

## 3. Utilisation
```bash
# tableau de bord web (Streamlit)
streamlit run src/dashboard_app.py

# réel (machine avec le checkpoint + GPU)
python src/haca_pipeline.py --srt data/raw/srt/8.srt --model marbertv2-haca
python src/haca_pipeline.py --srt-dir data/raw/srt --model marbertv2-haca \
       --out tonality.json --csv tonality.csv

# test de plomberie sans modèle (classifieur par mots-clés)
python src/haca_pipeline.py --srt-dir data/raw/srt --stub
```

**Analyse par locuteur (SRT diarisé).** Si le SRT est **diarisé** — chaque cue commence par une
étiquette `[SPEAKER_XX]`, comme produit par le pipeline de transcription (WhisperX + pyannote) —
le tableau de bord peut donner la tonalité **par locuteur**. Cocher *« Analyse par locuteur (SRT
diarisé) »* dans la barre latérale avant *« Lancer l'analyse »* : on obtient un tableau comparatif
(verdict + neg/neu/pos + couverture + confiance pour chaque `SPEAKER_XX`), un export CSV et une
barre de proportions par locuteur. L'étiquette de locuteur est retirée du texte avant la
classification (donc le verdict global de l'émission est aussi corrigé sur les fichiers diarisés),
et seuls les cues **consécutifs du même locuteur** sont fusionnés en énoncés. Détails techniques :
`report/PIPELINE.md` §4c. Détection : un SRT est considéré diarisé si ≥ 50 % des cues portent une
étiquette. Sans diarisation, l'option est sans effet (message d'information).

## 4. Sorties
- **Console** : par fichier, tonalité dominante + proportions + confiance + couverture + une
  **timeline** par segment (`▲pos ■neu ▼neg ·àrevoir`).
- **JSON** (`--out` / `--out-dir`) : émission + liste de segments, avec distribution, dominante,
  confiance, couverture, `flag_review` + raison.
- **CSV tableau de bord** (`--csv`) : une ligne par segment (et par émission) — importable
  directement dans Excel / Power BI : `dominant, p_neg, p_neu, p_pos, confidence, coverage,
  flag_review, reason`.

## 5. Comportement attendu / SLA réaliste
- **Ne pas viser 100 % d'automatisation.** Viser une **haute précision** sur les segments
  confiants + une **file de révision** pour les segments signalés (garbled, ambigus).
- Les émissions à dominante claire (ex. documentaire corruption → **neg**) sortent fiables ;
  les fichiers très bruités sortent avec **couverture basse + drapeau** (ex. 1.srt/6.srt) — ce
  qui est le bon comportement (on ne devine pas sur du bruit).
- La classe positive reste rare ; le système la signale quand elle est nette, sinon neutre.

## 6. Améliorations futures (par ordre d'impact)
1. **Meilleure ASR** (si la transcription devient contrôlable) : c'est le plus gros levier —
   le modèle échoue surtout sur ce qu'il ne peut pas lire.
2. **Escalade LLM** : l'option **API Cloud** (LLM via Chat Completions) permet déjà d'utiliser
   un LLM pour l'ensemble de la classification en un batch. Une prochaine étape serait le
   **second avis** : router uniquement les segments à faible confiance vers le LLM (Atlas-Chat
   en local ou API Cloud), pour économiser les appels API tout en bénéficiant de l'intelligence
   du LLM sur les cas ambigus.
3. **Évaluation au niveau segment** : annoter quelques émissions au niveau segment (humain)
   pour mesurer la qualité réelle du système — la métrique par énoncé sous-estime sa valeur.
4. **Calibrage des seuils de révision** (`COVERAGE_MIN`, `MAJORITY_MIN`, `CONF_MIN`) selon le
   taux de révision humaine acceptable par la HACA.
