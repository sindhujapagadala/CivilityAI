"""
CivilityAI: Unit Tests for Text Preprocessing & Normalization.
"""

import pytest
from safety_ml.text_processing import prepare_message_text, batch_prepare_messages


def test_empty_message_returns_empty_string():
    """Verifies empty string input produces safe empty normalized string."""
    assert prepare_message_text("") == ""
    assert prepare_message_text(None) == ""


def test_whitespace_only_message_is_collapsed():
    """Verifies whitespace-only message is reduced to empty string."""
    assert prepare_message_text("    \n\t   \r  ") == ""


def test_normal_civil_comment_preservation():
    """Verifies standard civil comments are correctly cleaned and lowercased."""
    raw = "Thank you for the constructive feedback on this article!"
    cleaned = prepare_message_text(raw)
    assert "constructive feedback" in cleaned
    assert cleaned == "thank you for the constructive feedback on this article!"


def test_url_redaction():
    """Verifies hyperlinked URLs are replaced with standard [URL] tokens."""
    raw = "Check this page: https://en.wikipedia.org/wiki/Peace and www.example.com"
    cleaned = prepare_message_text(raw)
    assert "https://" not in cleaned
    assert "www.example.com" not in cleaned
    assert "[url]" in cleaned.lower()


def test_email_redaction():
    """Verifies email addresses are replaced with standard [EMAIL] tokens."""
    raw = "Contact the admin at editor@civility.org for revisions."
    cleaned = prepare_message_text(raw)
    assert "editor@civility.org" not in cleaned
    assert "[email]" in cleaned.lower()


def test_html_entity_and_tag_stripping():
    """Verifies HTML markup and encoded entities are properly sanitized."""
    raw = "<div>Hello &amp; welcome &lt;editor&gt;!</div>"
    cleaned = prepare_message_text(raw)
    assert "<div>" not in cleaned
    assert "&amp;" not in cleaned
    assert "&lt;" not in cleaned
    assert "hello & welcome <editor>!" == cleaned


def test_unicode_and_emoji_handling():
    """Verifies unicode diacritics and emojis are handled without exceptions."""
    raw = "Café résumé 😃 👍🏽 with special chars: Ⓡ 𝓤𝓷𝓲𝓬𝓸𝓭𝓮"
    cleaned = prepare_message_text(raw)
    assert len(cleaned) > 0
    assert "café" in cleaned or "cafe" in cleaned


def test_extremely_long_text_processing():
    """Verifies preprocessing handles very large user submissions gracefully."""
    long_text = "This is a repetitive sentence. " * 500
    cleaned = prepare_message_text(long_text)
    assert len(cleaned) > 1000
    assert "repetitive sentence" in cleaned


def test_batch_prepare_messages():
    """Verifies vectorized list processing."""
    messages = ["First message", "https://spam.org link", "   Spaced   out   "]
    results = batch_prepare_messages(messages)
    assert len(results) == 3
    assert results[2] == "spaced out"
