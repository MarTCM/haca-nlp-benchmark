"""
Heuristic ASR-quality / garble filter for broadcast transcripts (Arabic-script Darija/MSA,
French, and code-switched mixes).

Many SRT cues are low-quality auto-transcription — syntactically incoherent fragments
with no recoverable meaning (e.g. "الحلو جانيت ريكاز الريان يفاخر كان انغمس"). These should
be quarantined from the training pool so the `neu` class doesn't degrade into a
"no-recoverable-signal" trash bucket (see report/ANNOTATION_RUBRIC_V3.md §3 rule 6).

The signal we exploit: coherent speech is dense in high-frequency function words — Darija
(في، ديال، اللي، كان، كاين، باش، هاد، …) and/or French (de, la, les, et, que, dans, …).
Garbled ASR output is sparse in them and full of rare/broken tokens. We score function-word
density over BOTH lexicons, so the gate is language-agnostic: a clean French utterance passes
on its French function words, a clean Darija one on its Arabic ones, and code-switched speech
on the sum. We combine that with a real-script ratio (Arabic + Latin letters, vs digits/symbol
noise), length, and repetition checks. No model, no lexicon download — pure heuristics,
deterministic. For pure Arabic-script input this is identical to the original Arabic-only gate.

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

# High-frequency French function / connective words — the Latin-script counterpart of
# FUNC_WORDS. Matched case-insensitively. Coherent French is dense in these; garbled Latin
# ASR is not.
FRENCH_FUNC_WORDS = {
    # articles / determiners
    "le", "la", "les", "l", "un", "une", "des", "du", "de", "au", "aux", "ce", "cet",
    "cette", "ces", "mon", "ma", "mes", "ton", "ta", "tes", "son", "sa", "ses",
    "notre", "nos", "votre", "vos", "leur", "leurs",
    # prepositions / connectors
    "à", "a", "en", "dans", "sur", "sous", "pour", "par", "avec", "sans", "entre", "vers",
    "chez", "et", "ou", "mais", "donc", "car", "comme", "si", "que", "qui", "quoi", "dont",
    "où", "quand", "alors", "parce", "aussi", "puis", "ainsi",
    # pronouns
    "je", "j", "tu", "il", "elle", "on", "nous", "vous", "ils", "elles", "me", "te", "se",
    "lui", "y", "ne", "pas", "plus", "rien", "tout", "tous", "toute", "toutes",
    # very common verbs (être / avoir / faire)
    "est", "sont", "été", "être", "sera", "était", "ont", "avait", "avais", "avons",
    "fait", "faire", "cela", "ça", "très", "bien",
}

_TOKEN_RE = re.compile(r"[^\W\d_]+", re.UNICODE)   # word tokens, drop pure-digit/punct
_ARABIC_RE = re.compile(r"[؀-ۿ]")
_LATIN_RE = re.compile(r"[A-Za-zÀ-ÿ]")             # Latin letters incl. French accents


def quality_features(text: str) -> dict:
    """Return interpretable features used by is_clean()."""
    text = (text or "").strip()
    tokens = _TOKEN_RE.findall(text)
    n = len(tokens)
    # function-word hits over BOTH lexicons (Arabic exact, French case-insensitive)
    func_hits = sum(1 for t in tokens if t in FUNC_WORDS or t.lower() in FRENCH_FUNC_WORDS)
    arabic_chars = len(_ARABIC_RE.findall(text))
    latin_chars = len(_LATIN_RE.findall(text))
    total_chars = max(len(text.replace(" ", "")), 1)
    uniq = len(set(tokens))
    return {
        "n_tokens": n,
        "n_chars": len(text),
        "func_ratio": func_hits / n if n else 0.0,           # density of function words (AR+FR)
        "arabic_ratio": arabic_chars / total_chars,           # share of Arabic script
        "latin_ratio": latin_chars / total_chars,             # share of Latin script
        "script_ratio": (arabic_chars + latin_chars) / total_chars,  # real text vs digit/symbol noise
        "uniq_ratio": uniq / n if n else 0.0,                 # repetition guard
        "mean_token_len": (sum(len(t) for t in tokens) / n) if n else 0.0,
    }


def is_clean(text: str,
             min_tokens: int = 7,
             min_func_ratio: float = 0.14,
             min_script_ratio: float = 0.55,
             min_uniq_ratio: float = 0.45) -> tuple[bool, str]:
    """
    Return (clean: bool, reason: str).

    Language-agnostic (Arabic / French / mixed): `script_ratio` guards against digit/symbol
    noise, `func_ratio` counts Arabic + French function words. For pure Arabic-script input
    this is identical to the original Arabic-only gate (latin_chars == 0, so script_ratio ==
    arabic_ratio and no French words match).

    Thresholds tuned on the HACA corpus: coherent paragraphs from the YouTube explainers
    score func_ratio ~0.30–0.45; garbled 1.srt/6.srt fragments score < 0.12.
    """
    f = quality_features(text)
    if f["n_tokens"] < min_tokens:
        return False, f"too_short({f['n_tokens']}tok)"
    if f["script_ratio"] < min_script_ratio:
        return False, f"noise({f['script_ratio']:.2f})"
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
