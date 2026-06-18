"""
Topic detection for the HACA dashboard.

Two backends, same interface `detect(text) -> str`:
  * load_atlas_topic()  — Atlas-Chat-2B (Moroccan-Darija LLM), prompts for the main topic.
                          Needs a GPU (4-bit / bitsandbytes); the accurate option.
  * load_keyword_topic()— keyword lookup over a HACA-relevant taxonomy. Instant, CPU-only,
                          for demos and machines without a GPU.

`text` should be a representative sample of the programme's clean utterances (the dashboard
passes the joined clean text).
"""

# HACA-relevant taxonomy (Darija/Arabic cue words per topic).
TOPIC_KEYWORDS = {
    "Corruption": ["فساد", "رشوة", "رشوه", "اختلاس", "نزاهة", "تضارب المصالح", "الغدر", "محسوبية"],
    "Santé": ["صحة", "صحه", "مستشفى", "طبيب", "مرض", "دواء", "علاج", "ممرض", "مستوصف", "تطبيب"],
    "Fiscalité": ["ضريبة", "ضرائب", "ضريبه", "جبائي", "قانون المالية", "المداخيل", "الدخل", "اقتطاع"],
    "Économie": ["اقتصاد", "استثمار", "نمو", "تشغيل", "بطالة", "صادرات", "تضخم", "المقاولة"],
    "Bourse / Finance": ["بورصة", "بورصه", "اسهم", "سهم", "سندات", "مستثمر", "تداول", "اكتتاب"],
    "Marchés publics": ["صفقات", "العمومية", "لابيل دوفر", "مرسوم", "عروض", "التوريدات"],
    "Politique": ["حكومة", "برلمان", "حزب", "معارضة", "ملتمس", "انتخابات", "وزير", "الأغلبية"],
    "Sahara / Diplomatie": ["الصحراء", "البوليزاريو", "الحكم الذاتي", "الامم المتحدة", "سيادة", "المسيرة"],
    "Histoire": ["تاريخ", "قرن", "امبراطورية", "سلطان", "استعمار", "الدولة", "ميلادية", "المرابطين"],
    "Religion": ["تصوف", "العارفين", "الحكمة", "الحضور", "الذكر", "سبحانه", "الصوفية"],
    "Sport": ["منتخب", "كاس", "فريق", "لاعب", "بطولة", "ميدالية", "تأهل", "نادي"],
    "Éducation": ["تعليم", "مدرسة", "التلاميذ", "الباكالوريا", "جامعة", "الطلبة", "التكوين"],
}


def load_keyword_topic():
    def detect(text: str) -> str:
        text = str(text)
        scores = {t: sum(text.count(w) for w in kws) for t, kws in TOPIC_KEYWORDS.items()}
        best = max(scores, key=scores.get)
        # top-2 if a clear second topic exists
        ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
        if ranked[0][1] == 0:
            return "Général"
        if len(ranked) > 1 and ranked[1][1] >= max(1, 0.6 * ranked[0][1]):
            return f"{ranked[0][0]} / {ranked[1][0]}"
        return best
    return detect


CATEGORIES = list(TOPIC_KEYWORDS) + ["Autre"]

# Map a free-form LLM answer (French or Arabic) back to a taxonomy category.
_ALIASES = {
    "Corruption": ["corruption", "فساد", "رشو", "اختلاس"],
    "Santé": ["sant", "صح", "طب", "مستشف", "health"],
    "Fiscalité": ["fiscal", "ضريب", "ضرائب", "impôt", "tax"],
    "Économie": ["econom", "اقتصاد", "نمو"],
    "Bourse / Finance": ["bourse", "financ", "مالي", "بورص", "اسهم", "stock"],
    "Marchés publics": ["march", "صفق", "عموم", "procurement", "appel"],
    "Politique": ["politiqu", "politic", "سياس", "حكوم", "برلم", "حزب"],
    "Sahara / Diplomatie": ["sahara", "صحراء", "diplo", "بوليزاريو"],
    "Histoire": ["histoir", "history", "تاريخ", "تاريخي"],
    "Religion": ["religi", "دين", "ديني", "تصوف", "روحان", "spirit", "islam"],
    "Sport": ["sport", "رياض", "كرة", "منتخب", "football"],
    "Éducation": ["éduc", "educ", "تعليم", "مدرس", "school"],
}


def snap_category(ans: str) -> str:
    """Snap a free-form answer to a known category (FR or AR); else return it cleaned."""
    a = (ans or "").strip().lower()
    if not a:
        return ""
    for cat, keys in _ALIASES.items():
        if any(k in a for k in keys):
            return cat
    return ans.strip()[:40]


TOPIC_PROMPT = (
    "صنّف الموضوع الرئيسي للنص التالي ضمن فئة واحدة فقط من هذه القائمة، "
    "وأجب بالضبط باسم الفئة بدون أي شرح:\n"
    + "، ".join(CATEGORIES) + "\n"
    "النص: {text}\n"
    "الفئة:"
)


def load_ollama_topic(model: str = "gemma2", host: str = "http://localhost:11434",
                      timeout: int = 240):
    """Topic detector via a LOCAL Ollama server — fast on integrated GPUs (GGUF).

    Run any 3-7B instruct model you have pulled, e.g.:  ollama pull qwen2.5:3b
    Needs `ollama serve` running. The first call loads the model (slow); `keep_alive`
    keeps it warm so subsequent calls are fast. Pass a SHORT sample as `text` (the topic
    only needs a representative excerpt — a long prefill is what makes it slow).
    """
    import re
    import requests

    def detect(text: str) -> str:
        prompt = TOPIC_PROMPT.format(text=str(text)[:1200])   # short prefill = fast
        r = requests.post(f"{host}/api/generate", json={
            "model": model, "prompt": prompt, "stream": False,
            "think": False,             # disable reasoning for "thinking" models (Qwen3 etc.)
            "keep_alive": "10m",        # keep the model warm
            "options": {"num_predict": 32, "temperature": 0, "num_ctx": 2048}},
            timeout=timeout)
        r.raise_for_status()
        raw = r.json().get("response", "") or ""
        # strip reasoning blocks (closed or cut-off by num_predict) from thinking models
        raw = re.sub(r"<think>.*?</think>", " ", raw, flags=re.S)
        raw = re.sub(r"<think>.*$", " ", raw, flags=re.S)
        lines = [ln.strip(" .،؛:-\"'*`") for ln in raw.splitlines()]
        lines = [ln for ln in lines if ln]
        return snap_category(lines[0]) if lines else "غير محدد"
    return detect


def load_atlas_topic(size: str = "2b"):
    """Atlas-Chat topic detector via transformers (4-bit, CUDA only). Lazy heavy imports."""
    import torch
    from atlas_chat import build_quantized_model, MODEL_IDS

    tok, model = build_quantized_model(MODEL_IDS[size])

    def detect(text: str) -> str:
        prompt = TOPIC_PROMPT.format(text=str(text)[:1200])
        inputs = tok(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=16, do_sample=False,
                                 pad_token_id=tok.eos_token_id)
        gen = tok.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        ans = gen.strip().splitlines()[0].strip(" .،؛:-\"'") if gen.strip() else ""
        return snap_category(ans) if ans else "غير محدد"
    return detect


if __name__ == "__main__":
    # quick keyword check on the source SRTs
    import glob, os, sys
    sys.path.insert(0, os.path.dirname(__file__))
    from build_haca_pool import utterances_for_file
    from extract_utterances import load_file
    det = load_keyword_topic()
    for path in sorted(glob.glob("data/raw/srt/*.srt")):
        fmt, cues = load_file(path)
        text = " ".join(utterances_for_file(fmt, cues))
        print(f"{os.path.basename(path):12s} -> {det(text)}")
