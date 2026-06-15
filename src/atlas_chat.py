"""
Step 5 — Atlas-Chat zero-shot sentiment (GATED model).

Requires:
  huggingface-cli login   (Gemma/MBZUAI license accepted)
  pip install bitsandbytes accelerate

Usage:
    python src/atlas_chat.py --model 2b
    python src/atlas_chat.py --model 9b   # only if GPU quota allows
"""

import argparse
import re
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from utils import set_seeds, SEED
from label_maps import ATLAS_CHAT_MAP, ATLAS_CHAT_NON_REPONSE, apply_map
from harness import evaluate_model, free_gpu

set_seeds()

MODEL_IDS = {
    "2b": "MBZUAI-Paris/Atlas-Chat-2B",
    "9b": "MBZUAI-Paris/Atlas-Chat-9B",
}

LANGS = ["darija_ar", "arabizi"]  # as per plan

PROMPT_TEMPLATE = (
    "أنت نظام تحليل مشاعر. صنّف النص التالي إلى فئة واحدة فقط من: "
    "positif, neutre, negatif.\n"
    "النص: {text}\n"
    "الإجابة (كلمة واحدة فقط):"
)

VALID_LABELS = set(ATLAS_CHAT_MAP.keys())
LABEL_RE = re.compile(r"\b(positif|neutre|negatif)\b", re.IGNORECASE)


def build_quantized_model(model_id: str):
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.float16,
    )
    return tokenizer, model


def predict_one(text: str, tokenizer, model) -> str:
    prompt = PROMPT_TEMPLATE.format(text=text[:300])  # truncate long inputs
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=10,
            do_sample=False,
            temperature=1.0,
            pad_token_id=tokenizer.eos_token_id,
        )
    generated = tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    m = LABEL_RE.search(generated)
    if m:
        return m.group(0).lower()
    return ATLAS_CHAT_NON_REPONSE


def run_atlas(model_size: str) -> None:
    model_id = MODEL_IDS[model_size]
    print(f"\n=== Atlas-Chat-{model_size.upper()} ({model_id}) [zero-shot] ===")
    print("  NOTE: GATED model — ensure huggingface-cli login is done.")

    tokenizer, model = build_quantized_model(model_id)

    for lang in LANGS:
        non_response_count = 0
        total = [0]

        def predict_fn(texts):
            preds = []
            for t in texts:
                raw = predict_one(t, tokenizer, model)
                total[0] += 1
                if raw == ATLAS_CHAT_NON_REPONSE:
                    non_response_count_ = non_response_count  # captured below
                    preds.append("neu")  # fallback for metric computation
                else:
                    try:
                        preds.append(apply_map(raw, ATLAS_CHAT_MAP))
                    except KeyError:
                        preds.append("neu")
            return preds

        # We need to count non-responses separately
        import pandas as pd, os
        data_path = os.path.join(
            os.path.dirname(__file__), "..", "data", "test_sets", f"{lang}.csv"
        )
        df = pd.read_csv(data_path)
        texts = df["text"].tolist()
        raw_preds = [predict_one(t, tokenizer, model) for t in texts]
        non_resp = sum(1 for r in raw_preds if r == ATLAS_CHAT_NON_REPONSE)
        non_resp_rate = non_resp / len(texts)

        canonical = []
        for r in raw_preds:
            if r == ATLAS_CHAT_NON_REPONSE:
                canonical.append("neu")
            else:
                try:
                    canonical.append(apply_map(r, ATLAS_CHAT_MAP))
                except KeyError:
                    canonical.append("neu")

        def _static_predict(texts_):
            return canonical[:len(texts_)]

        evaluate_model(
            f"atlas-chat-{model_size}", lang, _static_predict, model_obj=model,
            extra_meta={
                "non_response_rate": round(non_resp_rate, 4),
                "non_response_count": non_resp,
                "model_size": model_size,
                "quantization": "4bit-nf4",
            },
        )

    free_gpu(model)
    del model


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="2b", choices=["2b", "9b"])
    args = parser.parse_args()
    run_atlas(args.model)
