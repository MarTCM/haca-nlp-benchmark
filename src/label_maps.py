"""
Central label-mapping table — single source of truth.

All model raw labels -> {neg, neu, pos}.
Binary models with no neutral class are handled explicitly.
"""

# ── xlm-t ──────────────────────────────────────────────────────────────────
# id2label verified locally: {0: 'negative', 1: 'neutral', 2: 'positive'}
XLM_T_MAP = {
    "negative": "neg",
    "neutral":  "neu",
    "positive": "pos",
}

# ── camelbert-da ───────────────────────────────────────────────────────────
# id2label verified locally: {0: 'positive', 1: 'negative', 2: 'neutral'}
# Model is natively 3-class — threshold heuristic NOT needed.
# (Plan assumed binary; actual Hub model has a neutral head.)
CAMELBERT_DA_MAP = {
    "positive": "pos",
    "negative": "neg",
    "neutral":  "neu",
}
CAMELBERT_NEU_THRESHOLD = None  # model already predicts neutral natively

# ── distilcamembert ────────────────────────────────────────────────────────
# 5-star model: labels "1 star" … "5 stars"
# Mapping: 1-2* -> neg, 3* -> neu, 4-5* -> pos
DISTILCAMEMBERT_MAP = {
    "1 star":  "neg",
    "2 stars": "neg",
    "3 stars": "neu",
    "4 stars": "pos",
    "5 stars": "pos",
}

# ── fine-tuned encoders (trained by us with 3-class head) ──────────────────
# Labels set during Trainer training: {0: 'neg', 1: 'neu', 2: 'pos'}
FINETUNED_MAP = {
    "neg": "neg",
    "neu": "neu",
    "pos": "pos",
    "LABEL_0": "neg",
    "LABEL_1": "neu",
    "LABEL_2": "pos",
}

# ── atlas-chat (LLM zero-shot) ─────────────────────────────────────────────
# Closed-vocabulary prompt -> {positif, neutre, negatif}
# Out-of-vocab => NON_REPONSE (counted separately)
ATLAS_CHAT_MAP = {
    "positif":  "pos",
    "neutre":   "neu",
    "negatif":  "neg",
}
ATLAS_CHAT_NON_REPONSE = "NON_REPONSE"


def apply_map(raw_label: str, mapping: dict) -> str:
    """Return canonical label or raise KeyError on unknown raw label."""
    key = raw_label.strip().lower()
    for k, v in mapping.items():
        if k.lower() == key:
            return v
    raise KeyError(f"Unknown label '{raw_label}' not in mapping {list(mapping)}")
