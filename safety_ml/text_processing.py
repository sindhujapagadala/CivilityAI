"""
CivilityAI: Text Preprocessing & Sanitization Pipeline.

Provides robust text normalization for user-generated content,
handling whitespace anomalies, URL redaction, emoji expansion,
and character encodings.
"""

from __future__ import annotations

import html
import re
import unicodedata
from typing import List, Optional


# Pre-compiled regular expressions for performance
URL_PATTERN = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)
EMAIL_PATTERN = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b")
WHITESPACE_PATTERN = re.compile(r"\s+")
CONSECUTIVE_PUNCTUATION_PATTERN = re.compile(r"([!?.,;:]){3,}")
HTML_TAG_PATTERN = re.compile(r"<[^>]+>")


def prepare_message_text(
    raw_text: Optional[str],
    normalize_urls: bool = True,
    normalize_emails: bool = True,
    decode_html_entities: bool = True,
    collapse_whitespace: bool = True,
    preserve_case: bool = False,
) -> str:
    """
    Normalizes and cleans user-submitted content prior to tokenization or feature extraction.

    Args:
        raw_text: The incoming text string or None.
        normalize_urls: If True, replaces hyperlinks with a standard placeholder.
        normalize_emails: If True, replaces email addresses with a standard placeholder.
        decode_html_entities: If True, unescapes HTML entities like &amp; &lt;.
        collapse_whitespace: If True, converts consecutive whitespace characters to a single space.
        preserve_case: If True, retains original casing; otherwise converts to lowercase.

    Returns:
        A sanitized, normalized string ready for downstream model ingestion.
    """
    if raw_text is None:
        return ""

    if not isinstance(raw_text, str):
        raw_text = str(raw_text)

    # 1. Unicode normalization (NFKC normalizes compatible characters, ligatures, accents)
    sanitized_text = unicodedata.normalize("NFKC", raw_text)

    # 2. Decode HTML entities
    if decode_html_entities:
        sanitized_text = html.unescape(sanitized_text)

    # 3. Strip HTML tags
    sanitized_text = HTML_TAG_PATTERN.sub(" ", sanitized_text)

    # 4. Redact URLs to prevent vocabulary pollution
    if normalize_urls:
        sanitized_text = URL_PATTERN.sub(" [URL] ", sanitized_text)

    # 5. Redact Email addresses
    if normalize_emails:
        sanitized_text = EMAIL_PATTERN.sub(" [EMAIL] ", sanitized_text)

    # 6. Compress exaggerated punctuation (e.g., '??????' -> '???')
    sanitized_text = CONSECUTIVE_PUNCTUATION_PATTERN.sub(r"\1\1\1", sanitized_text)

    # 7. Case standardization
    if not preserve_case:
        sanitized_text = sanitized_text.lower()

    # 8. Collapse whitespace and strip margins
    if collapse_whitespace:
        sanitized_text = WHITESPACE_PATTERN.sub(" ", sanitized_text).strip()

    return sanitized_text


def batch_prepare_messages(
    message_bodies: List[str],
    normalize_urls: bool = True,
) -> List[str]:
    """
    Vectorized batch preprocessing across a collection of messages.
    """
    return [
        prepare_message_text(message, normalize_urls=normalize_urls)
        for message in message_bodies
    ]
