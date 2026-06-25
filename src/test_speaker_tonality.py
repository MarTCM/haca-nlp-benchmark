"""Tests for per-speaker tonality on diarized SRTs.

Covers srt_utils.split_speaker / is_diarized and the haca_pipeline per-speaker
path (segment_srt tag-stripping, segment_srt_by_speaker, process_file
by_speaker). Uses the keyword stub classifier so no model/GPU is needed.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
import srt_utils as su          # noqa: E402
import haca_pipeline as hp      # noqa: E402

# Clean utterances that pass asr_quality.is_clean (function-word dense).
AR = "في 2022 المبالغ اللي كانوا كيخلصوا المغاربه على قطاع الصحه من جيبهم كتوصل تقريبا 63 فالميه"
FR = "et la question est de savoir si les gens vont vraiment dans les hôpitaux pour se faire soigner"


def _write(path, cues):
    """cues: list of (start, end, text)."""
    blocks = []
    for i, (s, e, t) in enumerate(cues, start=1):
        blocks.append(f"{i}\n{s} --> {e}\n{t}")
    path.write_text("\n\n".join(blocks) + "\n", encoding="utf-8")
    return str(path)


def _diarized_srt(tmp_path):
    return _write(tmp_path / "diar.srt", [
        ("00:00:00,000", "00:00:05,000", f"[SPEAKER_00] {AR}"),
        ("00:00:05,000", "00:00:10,000", f"[SPEAKER_01] {FR}"),
        ("00:00:10,000", "00:00:15,000", f"[SPEAKER_00] {AR}"),
        ("00:00:15,000", "00:00:20,000", f"[SPEAKER_01] {FR}"),
    ])


def _plain_srt(tmp_path):
    return _write(tmp_path / "plain.srt", [
        ("00:00:00,000", "00:00:05,000", AR),
        ("00:00:05,000", "00:00:10,000", FR),
        ("00:00:10,000", "00:00:15,000", AR),
    ])


# ── split_speaker ────────────────────────────────────────────────────────────
def test_split_speaker_variants():
    assert su.split_speaker("[SPEAKER_00] bonjour") == ("SPEAKER_00", "bonjour")
    assert su.split_speaker("[speaker 3] salut") == ("SPEAKER_03", "salut")
    assert su.split_speaker("[SPK-1] hi") == ("SPEAKER_01", "hi")


def test_split_speaker_no_tag():
    assert su.split_speaker("juste du texte") == (None, "juste du texte")
    # ordinary bracketed notes are not treated as speakers
    assert su.split_speaker("[music] ...") == (None, "[music] ...")


# ── is_diarized ──────────────────────────────────────────────────────────────
def test_is_diarized():
    diar = [{"text": "[SPEAKER_00] a"}, {"text": "[SPEAKER_01] b"}, {"text": "c"}]
    assert su.is_diarized(diar) is True
    plain = [{"text": "a"}, {"text": "b"}, {"text": "[SPEAKER_00] c"}]
    assert su.is_diarized(plain) is False     # 1/3 < 0.50
    assert su.is_diarized([]) is False


# ── segment_srt strips tags ──────────────────────────────────────────────────
def test_segment_srt_strips_speaker_tags(tmp_path):
    _, rows = hp.segment_srt(_diarized_srt(tmp_path))
    assert rows, "expected at least one utterance"
    assert all("[SPEAKER" not in r["text"] for r in rows)
    assert all("SPEAKER_" not in r["text"] for r in rows)


# ── segment_srt_by_speaker ───────────────────────────────────────────────────
def test_segment_srt_by_speaker_groups(tmp_path):
    fmt, diarized, by_speaker = hp.segment_srt_by_speaker(_diarized_srt(tmp_path))
    assert diarized is True
    assert set(by_speaker) == {"SPEAKER_00", "SPEAKER_01"}
    # tags are stripped inside the grouped rows too
    for rows in by_speaker.values():
        assert rows
        assert all("SPEAKER_" not in r["text"] for r in rows)


def test_segment_srt_by_speaker_plain_is_empty(tmp_path):
    fmt, diarized, by_speaker = hp.segment_srt_by_speaker(_plain_srt(tmp_path))
    assert diarized is False
    assert by_speaker == {}


# ── process_file by_speaker ──────────────────────────────────────────────────
def test_process_file_by_speaker(tmp_path):
    predict_proba, thr = hp.load_stub()
    rep = hp.process_file(_diarized_srt(tmp_path), predict_proba, thr, by_speaker=True)
    assert rep["diarized"] is True
    assert set(rep["speakers"]) == {"SPEAKER_00", "SPEAKER_01"}
    for spk, agg in rep["speakers"].items():
        assert agg["tone"] in {"neg", "neu", "pos"}
        assert agg["n_utterances"] >= 1
        assert set(agg["proportions"]) == {"neg", "neu", "pos"}
    # programme-level report is still present and unaffected in shape
    assert rep["programme"]["tone"] in {"neg", "neu", "pos"}


def test_process_file_no_speaker_flag_omits_breakdown(tmp_path):
    predict_proba, thr = hp.load_stub()
    rep = hp.process_file(_diarized_srt(tmp_path), predict_proba, thr)
    assert "speakers" not in rep
    assert "diarized" not in rep


def test_process_file_by_speaker_plain_srt(tmp_path):
    predict_proba, thr = hp.load_stub()
    rep = hp.process_file(_plain_srt(tmp_path), predict_proba, thr, by_speaker=True)
    assert rep["diarized"] is False
    assert rep["speakers"] == {}
