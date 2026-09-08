"""
CivilityAI: Adversarial Robustness & Evasion Defense Package.
"""

from robustness.text_obfuscation import (
    generate_obfuscated_message,
    apply_leetspeak,
    apply_character_spacing,
    apply_punctuation_insertion,
    apply_character_substitution,
    apply_synthetic_misspelling,
)
from robustness.robustness_evaluation import (
    evaluate_robustness,
    compare_clean_and_obfuscated_performance,
)

__all__ = [
    "generate_obfuscated_message",
    "apply_leetspeak",
    "apply_character_spacing",
    "apply_punctuation_insertion",
    "apply_character_substitution",
    "apply_synthetic_misspelling",
    "evaluate_robustness",
    "compare_clean_and_obfuscated_performance",
]
