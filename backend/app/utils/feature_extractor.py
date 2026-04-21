"""Feature extraction for the ML priority classifier.

Converts raw ticket text into a fixed-length feature dict.
Every feature has a documented justification.
"""

import logging
import re
import string

logger = logging.getLogger(__name__)

_URGENCY_KEYWORDS = frozenset(
    ["refund", "broken", "cancel", "outage", "down", "not working",
     "urgent", "asap", "help", "fix", "error", "failed", "crash"]
)


def _count_syllables(word: str) -> int:
    """Estimate syllable count using vowel-group counting.

    Args:
        word: Single word (punctuation already stripped).

    Returns:
        Syllable count, minimum 1.
    """
    word = word.lower().strip(string.punctuation)
    if not word:
        return 0
    count = len(re.findall(r"[aeiouy]+", word))
    # Silent trailing 'e' adjustment
    if word.endswith("e") and count > 1:
        count -= 1
    return max(count, 1)


def _flesch_reading_ease(text: str) -> float:
    """Compute Flesch Reading Ease score without external dependencies.

    Formula: 206.835 − 1.015×(words/sentences) − 84.6×(syllables/words)
    Higher score = easier to read. Range roughly 0–100.

    Args:
        text: Input text.

    Returns:
        Flesch score, or 0.0 if text is too short to compute.
    """
    sentences = max(len(re.findall(r"[.!?]+", text)), 1)
    words = text.split()
    if not words:
        return 0.0
    syllables = sum(_count_syllables(w) for w in words)
    return 206.835 - 1.015 * (len(words) / sentences) - 84.6 * (syllables / len(words))


def extract_features(text: str) -> dict[str, float]:
    """Extract numeric features from ticket text.

    Features and their justification:
    - char_count: Longer messages often signal detailed complaints (urgency proxy).
    - word_count: Same rationale; word-level granularity.
    - exclamation_count: Emotional emphasis correlates with urgency.
    - question_count: Questions may indicate confusion but not always urgency.
    - caps_ratio: ALL-CAPS words signal shouting / strong emotion.
    - urgency_keyword_count: Direct urgency signal from domain vocabulary.
    - flesch_reading_ease: Simpler text (higher score) is often a quick complaint.
    - avg_word_length: Longer words may indicate technical or formal complaints.

    Args:
        text: Raw ticket text. Must not be empty.

    Returns:
        Ordered dict of feature_name → float value.

    Raises:
        ValueError: If text is empty.
    """
    if not text:
        raise ValueError("text must not be empty")

    words = text.split()
    word_count = len(words)
    caps_words = sum(1 for w in words if w.isupper() and len(w) > 1)
    urgency_hits = sum(1 for kw in _URGENCY_KEYWORDS if kw in text.lower())

    avg_word_len = (
        sum(len(w.strip(string.punctuation)) for w in words) / word_count
        if word_count > 0
        else 0.0
    )

    reading_ease = _flesch_reading_ease(text)

    features = {
        "char_count": float(len(text)),
        "word_count": float(word_count),
        "exclamation_count": float(text.count("!")),
        "question_count": float(text.count("?")),
        "caps_ratio": caps_words / word_count if word_count > 0 else 0.0,
        "urgency_keyword_count": float(urgency_hits),
        "flesch_reading_ease": reading_ease,
        "avg_word_length": avg_word_len,
    }

    logger.debug("Features extracted", extra={"features": features})
    return features
