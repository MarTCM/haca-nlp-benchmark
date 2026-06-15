"""
Step 1 smoke test — verifies the environment and the multilingual XLM-T pipeline
on three representative sentences (one per language family).
"""

import sys
from utils import set_seeds

set_seeds()

print("Python:", sys.version)

try:
    import torch
    print(f"PyTorch  : {torch.__version__}")
    print(f"CUDA     : {torch.cuda.is_available()} "
          f"({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU-only'})")
except ImportError as e:
    print(f"[WARN] torch not available: {e}")

try:
    import transformers
    print(f"Transformers: {transformers.__version__}")
except ImportError as e:
    print(f"[WARN] transformers not available: {e}")

print("\n--- Smoke test: cardiffnlp/twitter-xlm-roberta-base-sentiment ---")

MODEL_ID = "cardiffnlp/twitter-xlm-roberta-base-sentiment"

SENTENCES = [
    ("French",  "Ce film est absolument magnifique, j'ai adoré chaque instant !"),
    ("MSA",     "هذا الفيلم رائع جداً وأنا سعيد بمشاهدته."),
    ("Arabizi", "hada film zwine bzaf, 7bito ktir!"),
]

from transformers import pipeline, AutoConfig

device = 0 if (torch.cuda.is_available() if "torch" in sys.modules else False) else -1

# Inspect real id2label before building pipeline
cfg = AutoConfig.from_pretrained(MODEL_ID)
print(f"id2label : {cfg.id2label}\n")

classifier = pipeline(
    "text-classification",
    model=MODEL_ID,
    device=device,
    top_k=None,  # return scores for all labels (replaces deprecated return_all_scores)
)

for lang, text in SENTENCES:
    raw = classifier(text)
    # top_k=None on a single string: returns list[dict] or list[list[dict]]
    scores = raw[0] if isinstance(raw[0], dict) else raw[0]
    if isinstance(scores, dict):
        scores = raw  # flat list of dicts
    best = max(scores, key=lambda x: x["score"])
    print(f"[{lang}] {text[:60]}")
    print(f"  -> {best['label']} ({best['score']:.3f})")
    print()

print("Smoke test PASSED.")
