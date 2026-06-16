"""
Step 5 — Atlas-Chat zero-shot sentiment.

Usage:
    python src/atlas_chat.py --model 2b
    python src/atlas_chat.py --model 9b   # only if GPU quota allows
"""

import argparse
import os
import re
import statistics
import time

import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from utils import set_seeds, SEED
from label_maps import ATLAS_CHAT_MAP, ATLAS_CHAT_NON_REPONSE, apply_map
from harness import evaluate_model, free_gpu, WARMUP

set_seeds()

MODEL_IDS = {
    "2b": "MBZUAI-Paris/Atlas-Chat-2B",
    "9b": "MBZUAI-Paris/Atlas-Chat-9B",
}

LANGS = ["darija_ar", "arabizi"]

PROMPT_TEMPLATE = (
    "أنت نظام تحليل مشاعر. صنّف النص التالي إلى فئة واحدة فقط من: "
    "positif, neutre, negatif.\n"
    "النص: {text}\n"
    "الإجابة (كلمة واحدة فقط):"
)

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
    prompt = PROMPT_TEMPLATE.format(text=text[:300])
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=10,
            do_sample=False,
            temperature=1.0,
            pad_token_id=tokenizer.eos_token_id,
        )
    generated = tokenizer.decode(
        out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True
    )
    m = LABEL_RE.search(generated)
    return m.group(0).lower() if m else ATLAS_CHAT_NON_REPONSE


def run_atlas(model_size: str) -> None:
    model_id = MODEL_IDS[model_size]
    print(f"\n=== Atlas-Chat-{model_size.upper()} ({model_id}) [zero-shot] ===")

    tokenizer, model = build_quantized_model(model_id)

    for lang in LANGS:
        data_path = os.path.join(
            os.path.dirname(__file__), "..", "data", "test_sets", f"{lang}.csv"
        )
        df = pd.read_csv(data_path)
        texts = df["text"].tolist()

        # Warm-up (not timed, matches harness convention)
        print(f"  Warming up on {WARMUP} samples …")
        for t in texts[:WARMUP]:
            predict_one(t, tokenizer, model)

        # Timed inference — measure each call individually (generate() is sequential)
        print(f"  Running inference on {len(texts)} samples …")
        latencies_ms = []
        raw_preds = []
        for t in texts:
            t0 = time.perf_counter()
            raw = predict_one(t, tokenizer, model)
            latencies_ms.append((time.perf_counter() - t0) * 1000)
            raw_preds.append(raw)

        median_latency_ms = statistics.median(latencies_ms)
        non_resp = sum(1 for r in raw_preds if r == ATLAS_CHAT_NON_REPONSE)
        non_resp_rate = round(non_resp / len(texts), 4)

        canonical = []
        for r in raw_preds:
            if r == ATLAS_CHAT_NON_REPONSE:
                canonical.append("neu")
            else:
                try:
                    canonical.append(apply_map(r, ATLAS_CHAT_MAP))
                except KeyError:
                    canonical.append("neu")

        evaluate_model(
            f"atlas-chat-{model_size}",
            lang,
            lambda texts_: canonical[: len(texts_)],
            model_obj=model,
            extra_meta={
                "non_response_rate": non_resp_rate,
                "non_response_count": non_resp,
                "model_size": model_size,
                "quantization": "4bit-nf4",
            },
            latency_override_ms=median_latency_ms,
        )

    free_gpu(model)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="2b", choices=["2b", "9b"])
    args = parser.parse_args()
    run_atlas(args.model)
