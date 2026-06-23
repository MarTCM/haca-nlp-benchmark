"""Smoke tests for load_api_classifier — snap_sentiment, fallback, batched prompt."""

import os
import sys
from unittest.mock import MagicMock, patch

import pytest
import requests

sys.path.insert(0, os.path.dirname(__file__))
import haca_pipeline as hp


def test_api_classifier_signature():
    """load_api_classifier returns a (callable, dict) tuple."""
    predict_proba, thr = hp.load_api_classifier(api_key="test-key")
    assert callable(predict_proba)
    assert isinstance(thr, dict)
    assert set(thr.keys()) == {"neg", "neu", "pos"}


def test_empty_input():
    """predict_proba on empty input returns [] and fallback_rate=0."""
    predict_proba, _ = hp.load_api_classifier(api_key="test-key")
    result = predict_proba([])
    assert result == []
    assert predict_proba.fallback_rate == 0.0
    assert predict_proba.first_raw is None


@patch("requests.post")
def test_successful_batch(mock_post):
    """Well-formed API response parses correctly."""
    _mock_response(mock_post, "positif\nneutre\nnegatif")

    predict_proba, _ = hp.load_api_classifier(api_key="test-key")
    texts = ["bonjour", "au revoir", "merci"]
    result = predict_proba(texts)

    assert len(result) == 3
    assert result[0]["pos"] > 0.9
    assert result[1]["neu"] > 0.9
    assert result[2]["neg"] > 0.9
    assert predict_proba.fallback_rate == 0.0


@patch("requests.post")
def test_fallback_on_api_error(mock_post):
    """When the API call raises, all utterances get equal distribution."""
    mock_post.side_effect = requests.exceptions.ConnectionError("API unreachable")

    predict_proba, _ = hp.load_api_classifier(api_key="test-key")
    texts = ["a", "b"]
    result = predict_proba(texts)

    assert len(result) == 2
    for d in result:
        assert abs(d["neg"] - 0.33) < 0.01
        assert abs(d["neu"] - 0.34) < 0.01
        assert abs(d["pos"] - 0.33) < 0.01
    assert predict_proba.fallback_rate == 1.0


@patch("requests.post")
def test_http_error_status(mock_post):
    """HTTP 4xx/5xx triggers fallback (raise_for_status catches it)."""
    mock_response = MagicMock()
    mock_response.raise_for_status.side_effect = requests.exceptions.HTTPError("429")
    mock_post.return_value = mock_response

    predict_proba, _ = hp.load_api_classifier(api_key="test-key")
    texts = ["a", "b"]
    result = predict_proba(texts)

    assert predict_proba.fallback_rate == 1.0


@patch("requests.post")
def test_english_labels(mock_post):
    """English labels from the model map correctly."""
    _mock_response(mock_post, "positive\nneutral\nnegative")

    predict_proba, _ = hp.load_api_classifier(api_key="test-key")
    result = predict_proba(["t1", "t2", "t3"])
    assert result[0]["pos"] > 0.9
    assert result[1]["neu"] > 0.9
    assert result[2]["neg"] > 0.9


@patch("requests.post")
def test_arabic_labels(mock_post):
    """Arabic labels from the model map correctly."""
    _mock_response(mock_post, "إيجابي\nمحايد\nسلبي")

    predict_proba, _ = hp.load_api_classifier(api_key="test-key")
    result = predict_proba(["t1", "t2", "t3"])
    assert result[0]["pos"] > 0.9
    assert result[1]["neu"] > 0.9
    assert result[2]["neg"] > 0.9


@patch("requests.post")
def test_partial_fallback(mock_post):
    """Unparseable lines get fallback, but valid ones still parse.

    Response has 3 lines: 'positif' ✓, 'bogus' ✗ (skipped), 'negatif' ✓
    → out = [pos, neg] after loop → 1 fallback appended → result[2] is fallback.
    """
    _mock_response(mock_post, "positif\nbogus\nnegatif")

    predict_proba, _ = hp.load_api_classifier(api_key="test-key")
    texts = ["a", "b", "c"]
    result = predict_proba(texts)

    assert len(result) == 3
    assert result[0]["pos"] > 0.9
    assert result[1]["neg"] > 0.9
    assert abs(result[2]["neg"] - 0.33) < 0.01
    assert predict_proba.fallback_rate == pytest.approx(1 / 3)
    assert predict_proba.first_raw is not None


@patch("requests.post")
def test_fewer_labels_than_utterances(mock_post):
    """Fewer response lines than utterances → fallback for remaining.

    Response: 'positif\nnegatif' (2 lines for 3 utterances)
    → out = [pos, neg] after loop → 1 fallback appended → result[2] is fallback.
    """
    _mock_response(mock_post, "positif\nnegatif")

    predict_proba, _ = hp.load_api_classifier(api_key="test-key")
    texts = ["a", "b", "c"]
    result = predict_proba(texts)

    assert len(result) == 3
    assert result[0]["pos"] > 0.9
    assert result[1]["neg"] > 0.9
    assert abs(result[2]["neg"] - 0.33) < 0.01
    assert predict_proba.fallback_rate == pytest.approx(1 / 3)
    assert predict_proba.first_raw is not None


@patch("requests.post")
def test_include_topic(mock_post):
    """include_topic=True triggers topic extraction (two API calls: topic + batch)."""
    mock_post.side_effect = [
        _response("Économie"),          # topic call
        _response("positif\n" * 20),   # batch sentiment call
    ]

    predict_proba, _ = hp.load_api_classifier(
        api_key="test-key", include_topic=True)
    texts = ["a"] * 20
    result = predict_proba(texts)

    assert len(result) == 20
    assert predict_proba.fallback_rate == 0.0
    assert predict_proba.topic is not None
    assert isinstance(predict_proba.topic, str)


def _response(content: str) -> MagicMock:
    r = MagicMock()
    r.json.return_value = {"choices": [{"message": {"content": content}}]}
    return r


def _mock_response(mock_post, content: str):
    mock_post.return_value = _response(content)
