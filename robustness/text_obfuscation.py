"""
CivilityAI: Text Obfuscation & Adversarial Perturbation Utilities.

Generates realistic adversarial evasion tactics used by bad actors to bypass filters:
1. Leetspeak numeric replacement
2. Inter-character whitespace dilation
3. Punctuation symbol injection
4. Homoglyph and character substitutions
5. Typo and misspelling injection
"""

from __future__ import annotations

import random
from typing import List, Optional


# Leetspeak substitution mapping
LEETSPEAK_MAP = {
    "a": "@",
    "e": "3",
    "i": "1",
    "o": "0",
    "s": "$",
    "t": "7",
    "l": "1",
    "b": "8",
    "g": "9",
}

# Homoglyphs and visually similar unicode characters
HOMOGLYPH_MAP = {
    "a": "а",  # Cyrillic small letter a
    "e": "е",  # Cyrillic small letter ie
    "o": "о",  # Cyrillic small letter o
    "p": "р",  # Cyrillic small letter er
    "c": "с",  # Cyrillic small letter es
    "i": "і",  # Cyrillic small letter i
    "s": "ѕ",  # Cyrillic small letter dze
}

PUNCTUATION_INJECTIONS = [".", "_", "*", "!", "-", "/"]


def apply_leetspeak(text: str, probability: float = 0.60, rng: Optional[random.Random] = None) -> str:
    """
    Substitutes standard Latin characters with leetspeak symbols (e.g., idiot -> 1d10t).
    """
    rng = rng or random.Random()
    transformed_chars: List[str] = []
    for char in text:
        lower_char = char.lower()
        if lower_char in LEETSPEAK_MAP and rng.random() < probability:
            transformed_chars.append(LEETSPEAK_MAP[lower_char])
        else:
            transformed_chars.append(char)
    return "".join(transformed_chars)


def apply_character_spacing(text: str, word_probability: float = 0.50, rng: Optional[random.Random] = None) -> str:
    """
    Expands words with intra-character spacing (e.g., idiot -> i d i o t).
    """
    rng = rng or random.Random()
    words = text.split()
    spaced_words: List[str] = []

    for word in words:
        if len(word) > 3 and rng.random() < word_probability:
            spaced_words.append(" ".join(list(word)))
        else:
            spaced_words.append(word)

    return " ".join(spaced_words)


def apply_punctuation_insertion(text: str, word_probability: float = 0.50, rng: Optional[random.Random] = None) -> str:
    """
    Injects punctuation delimiters between letters (e.g., idiot -> i.d.i.o.t or id!ot).
    """
    rng = rng or random.Random()
    words = text.split()
    altered_words: List[str] = []

    for word in words:
        if len(word) > 3 and rng.random() < word_probability:
            sep = rng.choice(PUNCTUATION_INJECTIONS)
            if rng.random() < 0.5:
                # Full interleaving: i.d.i.o.t
                altered_words.append(sep.join(list(word)))
            else:
                # Mid-word injection: id!ot
                split_idx = len(word) // 2
                altered_words.append(word[:split_idx] + sep + word[split_idx:])
        else:
            altered_words.append(word)

    return " ".join(altered_words)


def apply_character_substitution(text: str, probability: float = 0.50, rng: Optional[random.Random] = None) -> str:
    """
    Substitutes Latin letters with lookalike Cyrillic homoglyphs.
    """
    rng = rng or random.Random()
    transformed_chars: List[str] = []
    for char in text:
        lower_char = char.lower()
        if lower_char in HOMOGLYPH_MAP and rng.random() < probability:
            transformed_chars.append(HOMOGLYPH_MAP[lower_char])
        else:
            transformed_chars.append(char)
    return "".join(transformed_chars)


def apply_synthetic_misspelling(text: str, word_probability: float = 0.40, rng: Optional[random.Random] = None) -> str:
    """
    Simulates typos: swaps adjacent letters or drops repeated vowels.
    """
    rng = rng or random.Random()
    words = text.split()
    altered_words: List[str] = []

    for word in words:
        if len(word) > 4 and rng.random() < word_probability:
            swap_idx = rng.randint(1, len(word) - 2)
            char_list = list(word)
            char_list[swap_idx], char_list[swap_idx + 1] = char_list[swap_idx + 1], char_list[swap_idx]
            altered_words.append("".join(char_list))
        else:
            altered_words.append(word)

    return " ".join(altered_words)


def generate_obfuscated_message(
    message_body: str,
    strategy: str = "composite",
    rng_seed: Optional[int] = None,
) -> str:
    """
    Applies one or more obfuscation techniques to a message string.

    Strategies:
        'leetspeak': Number replacements
        'spacing': Character spacing
        'punctuation': Symbol injections
        'homoglyphs': Visual character substitutions
        'misspelling': Typo simulation
        'composite': Stochastically mixes all strategies
    """
    rng = random.Random(rng_seed)

    if strategy == "leetspeak":
        return apply_leetspeak(message_body, rng=rng)
    elif strategy == "spacing":
        return apply_character_spacing(message_body, rng=rng)
    elif strategy == "punctuation":
        return apply_punctuation_insertion(message_body, rng=rng)
    elif strategy == "homoglyphs":
        return apply_character_substitution(message_body, rng=rng)
    elif strategy == "misspelling":
        return apply_synthetic_misspelling(message_body, rng=rng)
    elif strategy == "composite":
        # Chain 2-3 random transformations
        result = message_body
        methods = [
            lambda t: apply_leetspeak(t, probability=0.35, rng=rng),
            lambda t: apply_punctuation_insertion(t, word_probability=0.35, rng=rng),
            lambda t: apply_character_substitution(t, probability=0.35, rng=rng),
            lambda t: apply_character_spacing(t, word_probability=0.30, rng=rng),
        ]
        selected_methods = rng.sample(methods, k=2)
        for method in selected_methods:
            result = method(result)
        return result
    else:
        return message_body
