"""
Heuristic ASR-quality / garble filter for Moroccan Darija (Arabic script).

Many SRT cues are low-quality auto-transcription — syntactically incoherent fragments
with no recoverable meaning (e.g. "الحلو جانيت ريكاز الريان يفاخر كان انغمس"). These should
be quarantined from the training pool so the `neu` class doesn't degrade into a
"no-recoverable-signal" trash bucket (see report/ANNOTATION_RUBRIC_V3.md §3 rule 6).

The signal we exploit: coherent Darija is dense in high-frequency function words
(في، ديال، اللي، كان، كاين، باش، هاد، …). Garbled ASR output is sparse in them and full of
rare/broken tokens. We combine that with length and repetition checks. No model, no lexicon
download — pure heuristics, deterministic.

Usage:
    from asr_quality import is_clean, quality_features
    ok, reason = is_clean(text)
"""

import re

# High-frequency Moroccan Darija + MSA function / connective words. Coherent broadcast
# speech is dense in these; garbled ASR fragments are not.
FUNC_WORDS = {
    # prepositions / particles
    "في", "من", "على", "الى", "إلى", "عن", "مع", "بين", "عند", "حتى", "بعد", "قبل",
    # demonstratives / relatives
    "هذا", "هذه", "هاد", "هادي", "هادو", "داك", "ديك", "دوك", "اللي", "الي", "اللى",
    # possessive ديال family
    "ديال", "ديالي", "ديالك", "ديالو", "ديالها", "ديالهم", "ديالنا",
    # verbs of being / future / existence
    "كان", "كانت", "كانوا", "غادي", "غدي", "كاين", "كاينة", "كاينين", "تيكون", "كيكون",
    "كنت", "ولى", "ولات", "بقى", "بقات",
    # pronouns
    "هو", "هي", "هما", "حنا", "انا", "أنا", "انت", "نتا", "نتي", "هاذ",
    # connectors / quantifiers
    "ولكن", "ماشي", "ماكاين", "كل", "كثير", "بزاف", "شي", "ولا", "اولا", "أولا",
    "يعني", "مثلا", "راه", "رانا", "راها", "باش", "علاش", "شنو", "كيفاش", "فين", "منين",
    "واحد", "وحدة", "دابا", "هكذا", "هكا", "زعما", "بحال", "كيما", "كيف",
    # very common content anchors in this corpus
    "الناس", "الدولة", "الدوله", "الفلوس", "خاص", "خصنا", "خصك", "نقول", "قال", "قالت",
    "عندي", "عندك", "عندو", "عندها", "عندنا",
}

_TOKEN_RE = re.compile(r"[^\W\d_]+", re.UNICODE)   # word tokens, drop pure-digit/punct
_ARABIC_RE = re.compile(r"[؀-ۿ]")


def quality_features(text: str) -> dict:
    """Return interpretable features used by is_clean()."""
    text = (text or "").strip()
    tokens = _TOKEN_RE.findall(text)
    n = len(tokens)
    func_hits = sum(1 for t in tokens if t in FUNC_WORDS)
    arabic_chars = len(_ARABIC_RE.findall(text))
    total_chars = max(len(text.replace(" ", "")), 1)
    uniq = len(set(tokens))
    return {
        "n_tokens": n,
        "n_chars": len(text),
        "func_ratio": func_hits / n if n else 0.0,       # density of function words
        "arabic_ratio": arabic_chars / total_chars,       # is it actually Arabic script
        "uniq_ratio": uniq / n if n else 0.0,             # repetition guard
        "mean_token_len": (sum(len(t) for t in tokens) / n) if n else 0.0,
    }


def is_clean(text: str,
             min_tokens: int = 7,
             min_func_ratio: float = 0.14,
             min_arabic_ratio: float = 0.55,
             min_uniq_ratio: float = 0.45) -> tuple[bool, str]:
    """
    Return (clean: bool, reason: str).

    Thresholds tuned on the HACA corpus: coherent paragraphs from the YouTube explainers
    score func_ratio ~0.30–0.45; garbled 1.srt/6.srt fragments score < 0.12.
    """
    f = quality_features(text)
    if f["n_tokens"] < min_tokens:
        return False, f"too_short({f['n_tokens']}tok)"
    if f["arabic_ratio"] < min_arabic_ratio:
        return False, f"low_arabic({f['arabic_ratio']:.2f})"
    if f["uniq_ratio"] < min_uniq_ratio:
        return False, f"repetitive({f['uniq_ratio']:.2f})"
    if f["func_ratio"] < min_func_ratio:
        return False, f"garbled_low_func({f['func_ratio']:.2f})"
    return True, "clean"


if __name__ == "__main__":
    # Quick self-check on known-good and known-garbled samples.
    samples = [
        ("في 2022 المبالغ اللي كانوا كيخلصوا المغاربه على قطاع الصحه من جيبهم كتوصل تقريبا 63 فالميه", "expect clean"),
        ("الحلو جانيت ريكاز الريان يفاخر كان انغمس", "expect garbled"),
        ("كولومبيا يمني صغير الضغط المتفق تنام مورت", "expect garbled"),
        ("ولكن السؤال اشنا هما المشاكل وباش نعرفوهم غادي ندوزوا للمحور الثالث ديال المنظومه الصحيه", "expect clean"),
        ("ما ينكسر", "expect garbled/short"),
    ]
    for txt, note in samples:
        ok, why = is_clean(txt)
        f = quality_features(txt)
        print(f"[{'CLEAN' if ok else 'DROP '}] {why:22s} func={f['func_ratio']:.2f} "
              f"ntok={f['n_tokens']:2d}  {note}")
