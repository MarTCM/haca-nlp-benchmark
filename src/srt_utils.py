"""
SRT pipeline utilities.

- parse_srt()       : parse an SRT file into cue dicts
- cues_to_utterances: merge cues into sentence-level utterances
- detect_lang()     : script heuristic router (Arabic / arabizi / francais)
- validate_router() : confusion matrix of the router on a hand-checked sample
"""

import html
import re
import unicodedata
from typing import List, Dict, Optional, Tuple

ARABIC_RE   = re.compile(r"[؀-ۿ]")
ARABIZI_RE  = re.compile(r"\b\w*[379]\w*\b")  # digits 3/7/9 typical of Arabizi
SENTENCE_END = re.compile(r"(?<=[.!?؟۔])\s+")

# Diarized SRTs (WhisperX + pyannote, via the transcription pipeline) prefix each
# cue's text with a speaker tag, e.g. ``[SPEAKER_00] ...``. Match that leading tag
# (and close variants like ``[SPEAKER 0]`` / ``[SPK_1]``) so per-speaker analysis
# can split it off; ordinary bracketed notes like ``[music]`` are deliberately not
# matched.
SPEAKER_RE = re.compile(r"^\s*\[\s*(SPEAKER|SPK|SPEAKER_ID|S)[\s_-]?(\d+)\s*\]\s*",
                        re.IGNORECASE)


def split_speaker(text: str) -> Tuple[Optional[str], str]:
    """Split a leading diarization speaker tag from a cue's text.

    Returns ``(speaker_label, remaining_text)``; ``(None, stripped_text)`` when
    there is no speaker tag. The label is normalised to ``SPEAKER_<NN>`` form so
    ``[SPEAKER_0]``, ``[speaker 0]`` and ``[SPK-0]`` all map to ``SPEAKER_00``.
    """
    text = text or ""
    m = SPEAKER_RE.match(text)
    if not m:
        return None, text.strip()
    label = f"SPEAKER_{int(m.group(2)):02d}"
    return label, text[m.end():].strip()


def is_diarized(cues: List[Dict], min_fraction: float = 0.50) -> bool:
    """True if at least ``min_fraction`` of cues carry a ``[SPEAKER_XX]`` tag.

    A genuinely diarized SRT (WhisperX + pyannote) tags nearly every cue, so the
    0.5 default comfortably separates it from a plain SRT that happens to contain
    a stray bracketed note.
    """
    if not cues:
        return False
    tagged = sum(1 for c in cues if split_speaker(c.get("text", ""))[0] is not None)
    return tagged / len(cues) >= min_fraction

_ENCODINGS = ("utf-8", "utf-8-sig", "cp1256", "latin-1")


def _strip_html(text: str) -> str:
    text = re.sub(r"<[^>]+>", "", text)
    return html.unescape(text).strip()


def parse_srt(path: str) -> List[Dict]:
    """Return list of {index, start, end, text} dicts."""
    content = None
    for enc in _ENCODINGS:
        try:
            with open(path, encoding=enc) as fh:
                content = fh.read()
            break
        except (UnicodeDecodeError, LookupError):
            continue
    if content is None:
        raise ValueError(f"Could not decode {path} with any of {_ENCODINGS}")

    try:
        import pysrt
        subs = pysrt.open(path)
        return [
            {"index": s.index, "start": str(s.start), "end": str(s.end),
             "text": _strip_html(s.text)}
            for s in subs
        ]
    except Exception:
        # Fallback: manual regex parse
        block_re = re.compile(
            r"(\d+)\s*\n(\d{2}:\d{2}:\d{2}[,\.]\d{3})\s*-->\s*(\d{2}:\d{2}:\d{2}[,\.]\d{3})\s*\n([\s\S]*?)(?=\n\n|\Z)",
            re.MULTILINE,
        )
        cues = []
        for m in block_re.finditer(content):
            cues.append({
                "index": int(m.group(1)),
                "start": m.group(2),
                "end":   m.group(3),
                "text":  _strip_html(m.group(4).replace("\n", " ").strip()),
            })
        return cues


def cues_to_utterances(cues: List[Dict]) -> List[str]:
    """Merge cue texts into utterances split on sentence boundaries."""
    raw = " ".join(c["text"] for c in cues if c["text"])
    parts = SENTENCE_END.split(raw)
    return [p.strip() for p in parts if p.strip()]


def detect_lang(text: str) -> str:
    """
    Heuristic language router.

    Returns one of: 'arabe', 'arabizi', 'francais'.
    MSA vs Darija disambiguation is delegated to a downstream camel-tools call.
    """
    if ARABIC_RE.search(text):
        return "arabe"
    tokens = re.findall(r"\w+", text)
    arabizi_hits = sum(1 for t in tokens if ARABIZI_RE.match(t)) if tokens else 0
    if arabizi_hits / max(len(tokens), 1) > 0.10:
        return "arabizi"
    return "francais"


def validate_router(
    hand_checked: List[Tuple[str, str]],
) -> None:
    """
    Compute and print a confusion matrix of the router.

    Parameters
    ----------
    hand_checked : list of (text, true_lang) with true_lang in
                   {arabe, arabizi, francais}
    """
    from sklearn.metrics import classification_report, confusion_matrix

    y_true = [lang for _, lang in hand_checked]
    y_pred = [detect_lang(text) for text, _ in hand_checked]
    classes = sorted(set(y_true + y_pred))

    print("=== Language Router Validation ===")
    print(classification_report(y_true, y_pred, labels=classes, zero_division=0))
    cm = confusion_matrix(y_true, y_pred, labels=classes)
    print("Confusion matrix (rows=true, cols=pred):")
    print("Labels:", classes)
    print(cm)
