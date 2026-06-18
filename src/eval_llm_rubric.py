"""
Stage 6 — Rubric-prompted LLM classifier on the broadcast gold test.

The content-valence task (judge what's *described*, not the anchor's tone) is a
rubric-following task, which favours an instructable LLM you can hand the rubric to over a
small encoder that must infer it from a few hundred examples.  This script prompts
Atlas-Chat (Moroccan-Darija LLM) with Rubric v3 + few-shot exemplars and evaluates on the
frozen gold `domaine_reel_v2`, through the same harness as every other model.

Few-shot exemplars are illustrative paraphrases (NOT drawn from the test set).

Usage:
    python src/eval_llm_rubric.py --model 2b
    python src/eval_llm_rubric.py --model 9b     # if GPU quota allows
"""

import argparse
import os
import re
import statistics
import time

import pandas as pd
import torch

from utils import set_seeds
from label_maps import apply_map, ATLAS_CHAT_MAP
from harness import evaluate_model, free_gpu, WARMUP
from atlas_chat import build_quantized_model, MODEL_IDS

set_seeds()

GOLD = "data/test_sets/domaine_reel_v2.csv"
LABEL_RE = re.compile(r"\b(positif|neutre|negatif)\b", re.IGNORECASE)

# Rubric v3 (content-valence) + few-shot, in Arabic. The model is TOLD the rule.
RUBRIC_PROMPT = (
    "أنت محلل مضمون إعلامي للهيئة العليا للاتصال السمعي البصري. "
    "صنّف المحتوى حسب ما يصفه، وليس حسب نبرة المذيع:\n"
    "- negatif: المحتوى يصف شيئا سيئا (فشل، فساد، نقص، أزمة، خسارة، ضرر، هجرة الأطباء).\n"
    "- positif: المحتوى يصف شيئا جيدا (نجاح، إصلاح ينفع الناس، تقدم، نمو، فرصة، إنجاز).\n"
    "- neutre: محتوى إجرائي أو تعريفي أو وصفي بدون حكم واضح، حتى لو كان الموضوع حساسا؛ "
    "أو نص غير مفهوم.\n\n"
    "أمثلة:\n"
    "النص: مليار ونصف اختلست من صندوق الضمان الاجتماعي. الإجابة: negatif\n"
    "النص: الأسهم تضاعفت سبع مرات والمستثمرون ربحوا أموالا كثيرة. الإجابة: positif\n"
    "النص: جدول الضريبة على الدخل فيه هذه الشرائح من صفر إلى ثمانية وثلاثين بالمئة. الإجابة: neutre\n"
    "النص: الأطباء يهاجرون وقطاع الصحة يعاني من خصاص كبير. الإجابة: negatif\n"
    "النص: الإصلاح الجديد خفف الضريبة على الطبقة المتوسطة ونفعها. الإجابة: positif\n"
    "النص: المنظومة الصحية تتكون من ثلاث ركائز أساسية. الإجابة: neutre\n\n"
    "النص: {text}\n"
    "الإجابة (كلمة واحدة فقط، positif أو neutre أو negatif):"
)


def predict_one(text, tokenizer, model):
    prompt = RUBRIC_PROMPT.format(text=str(text)[:400])
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=8, do_sample=False,
                             temperature=1.0, pad_token_id=tokenizer.eos_token_id)
    gen = tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    m = LABEL_RE.search(gen)
    return m.group(0).lower() if m else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="2b", choices=list(MODEL_IDS))
    args = ap.parse_args()

    model_id = MODEL_IDS[args.model]
    print(f"\n=== Atlas-Chat-{args.model.upper()} [rubric-prompted] on {GOLD} ===")
    tokenizer, model = build_quantized_model(model_id)

    df = pd.read_csv(GOLD)
    texts = df["text"].tolist()

    for t in texts[:WARMUP]:
        predict_one(t, tokenizer, model)

    latencies, preds = [], []
    non_resp = 0
    for t in texts:
        t0 = time.perf_counter()
        raw = predict_one(t, tokenizer, model)
        latencies.append((time.perf_counter() - t0) * 1000)
        if raw is None:
            non_resp += 1
            preds.append("neu")
        else:
            try:
                preds.append(apply_map(raw, ATLAS_CHAT_MAP))
            except KeyError:
                preds.append("neu")

    evaluate_model(
        f"atlas-chat-{args.model}-rubric", "domaine_reel_v2",
        lambda texts_: preds[: len(texts_)],
        model_obj=model,
        extra_meta={"prompt": "rubric-v3+fewshot",
                    "non_response_rate": round(non_resp / len(texts), 4),
                    "model_size": args.model, "quantization": "4bit-nf4"},
        latency_override_ms=statistics.median(latencies),
    )
    free_gpu(model)


if __name__ == "__main__":
    main()
