"""
CivilityAI: Unit Tests for Adversarial Robustness & Obfuscation Transforms.
"""

import random
import pytest
from robustness.text_obfuscation import (
    apply_character_spacing,
    apply_character_substitution,
    apply_leetspeak,
    apply_punctuation_insertion,
    apply_synthetic_misspelling,
    generate_obfuscated_message,
)


def test_leetspeak_substitution():
    """Verifies leetspeak character conversion."""
    rng = random.Random(42)
    original = "idiot attack"
    obfuscated = apply_leetspeak(original, probability=1.0, rng=rng)
    # Characters like i, o, a, t should be replaced with numbers/symbols
    assert obfuscated != original
    assert any(c in obfuscated for c in ["1", "0", "@", "7"])


def test_character_spacing_expansion():
    """Verifies intra-word spacing insertion."""
    rng = random.Random(42)
    original = "violent idiot threat"
    spaced = apply_character_spacing(original, word_probability=1.0, rng=rng)
    assert len(spaced) > len(original)
    # Space separated letters
    assert "i d i o t" in spaced or "v i o l e n t" in spaced


def test_punctuation_insertion():
    """Verifies symbol delimiters are inserted into words."""
    rng = random.Random(42)
    original = "abusive toxic behavior"
    punctuated = apply_punctuation_insertion(original, word_probability=1.0, rng=rng)
    assert punctuated != original
    assert any(sym in punctuated for sym in [".", "_", "*", "!", "-", "/"])


def test_homoglyph_substitution():
    """Verifies visually identical Cyrillic homoglyphs replace Latin letters."""
    rng = random.Random(42)
    original = "parasite attack"
    homoglyphed = apply_character_substitution(original, probability=1.0, rng=rng)
    assert homoglyphed != original


def test_composite_obfuscation_generator():
    """Verifies composite perturbation strategy generates realistic adversarial text."""
    original = "Shut up and die, you pathetic loser."
    obfuscated = generate_obfuscated_message(original, strategy="composite", rng_seed=99)
    assert len(obfuscated) > 0
    assert obfuscated != original
