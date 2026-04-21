"""Tests for app.utils.feature_extractor.

No external dependencies — pure unit tests.
"""

import pytest

from app.utils.feature_extractor import extract_features


def test_extract_features_returns_all_keys():
    """extract_features returns all expected feature keys."""
    features = extract_features("My account is broken!!!")
    expected_keys = {
        "char_count", "word_count", "exclamation_count", "question_count",
        "caps_ratio", "urgency_keyword_count", "flesch_reading_ease", "avg_word_length",
    }
    assert set(features.keys()) == expected_keys


def test_extract_features_exclamation_count():
    """Exclamation marks are counted correctly."""
    features = extract_features("Help!!! URGENT!!!")
    assert features["exclamation_count"] == 6.0


def test_extract_features_urgency_keyword_detected():
    """Urgency keywords increment the urgency_keyword_count."""
    features = extract_features("I need a refund asap")
    assert features["urgency_keyword_count"] >= 2.0


def test_extract_features_no_urgency_keywords():
    """Ticket with no urgency vocabulary has zero urgency keyword count."""
    features = extract_features("Your order has been shipped successfully")
    assert features["urgency_keyword_count"] == 0.0


def test_extract_features_caps_ratio():
    """ALL-CAPS words are detected and ratio is computed correctly."""
    features = extract_features("THIS IS URGENT please fix")
    # 3 caps words out of 5 total = 0.6
    assert features["caps_ratio"] == pytest.approx(0.6, abs=0.01)


def test_extract_features_empty_raises():
    """Empty text raises ValueError."""
    with pytest.raises(ValueError, match="must not be empty"):
        extract_features("")


def test_extract_features_all_floats():
    """All feature values are floats."""
    features = extract_features("My order is broken and I need help")
    assert all(isinstance(v, float) for v in features.values())
