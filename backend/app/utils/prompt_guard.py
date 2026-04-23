"""Prompt injection guard for all LLM-bound user input.

Enforces CLAUDE.md §19 rules:
- Sanitize before format(): replace characters that break XML-tag delimiters
- Log suspicious injection patterns; never silently reject (§19 rule 6)
"""

import logging
import re
import unicodedata

logger = logging.getLogger(__name__)

# Patterns that suggest deliberate prompt override attempts — log as warning, do not reject
_INJECTION_RE = re.compile(
    r"(</?\s*(user_input|retrieved_context|system|instruction)\b"
    r"|ignore\s+(previous|all)\s+(instructions?|above)"
    r"|new\s+instruction"
    r"|forget\s+(everything|all|above|instructions?)"
    r"|you\s+are\s+now\b"
    r"|act\s+as\b"
    r"|disregard\s+(all|previous))",
    re.IGNORECASE,
)


def sanitize_user_input(text: str) -> str:
    """Strip prompt-injection vectors from user-supplied text.

    Replaces angle brackets so XML-tag delimiters in prompt templates
    cannot be closed or overridden by user content. Strips null bytes
    and non-printable control characters (preserves newlines and tabs).
    Suspicious override phrases are logged as warnings but NOT rejected —
    observability over censorship (CLAUDE.md §19 rule 6).

    Args:
        text: Raw user-supplied string from the API request.

    Returns:
        Sanitized string safe to interpolate into prompt templates.
    """
    # Break XML tag injection — replace < > with Unicode angle brackets
    cleaned = text.replace("<", "⟨").replace(">", "⟩")

    # Strip control characters except \n and \t
    cleaned = "".join(
        ch
        for ch in cleaned
        if ch in ("\n", "\t") or not unicodedata.category(ch).startswith("C")
    )

    if _INJECTION_RE.search(text):
        logger.warning(
            "Potential prompt injection detected",
            extra={"length": len(text), "preview": text[:120]},
        )

    return cleaned
