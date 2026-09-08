"""
CivilityAI: Synthetic Benchmark Dataset Generator.

Generates realistic, representative multi-label text moderation datasets
adhering strictly to the schema of the Jigsaw Toxic Comment Classification Challenge.
This enables complete pipeline verification, exploratory analysis, model training,
and testing without requiring immediate manual Kaggle downloads.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import random
from pathlib import Path
from typing import Dict, List, Tuple


# Canonical dataset raw columns matching Jigsaw competition specification
CANONICAL_RAW_COLUMNS = [
    "id",
    "comment_text",
    "toxic",
    "severe_toxic",
    "obscene",
    "threat",
    "insult",
    "identity_hate",
]

# Curated seed phrases across categories for synthetic synthesis
CIVIL_MESSAGE_TEMPLATES = [
    "Thank you for improving this article, the references look very solid.",
    "Could we discuss the proposed changes on the talk page before reverting?",
    "I disagree with this interpretation, but appreciate your detailed explanation.",
    "The neutral point of view policy suggests we should include both historical perspectives.",
    "Please check section 3; there appears to be a typo in the citation year.",
    "Great work summarizing the recent scientific consensus on this topic.",
    "I have added secondary sources to corroborate the claims made in the lead section.",
    "Would anyone object if I restructured the taxonomy section for clarity?",
    "Let us keep this discussion focused on verifiable facts and primary references.",
    "I restored the previous revision because the added paragraph lacked reliable citations.",
    "Thanks for the heads up regarding the copyright status of that image.",
    "I think we need additional peer-reviewed articles to support this controversial claim.",
    "The formatting is now aligned with the community style guidelines.",
    "Appreciate your patience while we reach consensus on this dispute.",
    "This edit introduces good context, though we might trim the introductory summary.",
    "Can you provide a link to the original archival document referenced here?",
    "Welcome to the community! Feel free to ask if you need help formatting citations.",
    "The neutrality dispute has been resolved following the rewrite of paragraph two.",
    "I believe both viewpoints can be integrated fairly with balanced prose.",
    "Let us refrain from making assumptions and verify against the provided bibliography.",
]

TOXIC_MESSAGE_TEMPLATES: List[Tuple[str, Dict[str, int]]] = [
    # (template, {category: flag})
    # General toxicity + Insult
    (
        "You are completely incompetent and have no idea what you are talking about.",
        {"toxic": 1, "severe_toxic": 0, "obscene": 0, "threat": 0, "insult": 1, "identity_hate": 0},
    ),
    (
        "Stop editing this page you absolute moron. Nobody wants your garbage here.",
        {"toxic": 1, "severe_toxic": 0, "obscene": 0, "threat": 0, "insult": 1, "identity_hate": 0},
    ),
    (
        "Why do idiots like you even bother commenting when you clearly know nothing?",
        {"toxic": 1, "severe_toxic": 0, "obscene": 0, "threat": 0, "insult": 1, "identity_hate": 0},
    ),
    # Obscenity + Toxicity + Insult
    (
        "Shut the fuck up and delete your ridiculous account, piece of shit.",
        {"toxic": 1, "severe_toxic": 0, "obscene": 1, "threat": 0, "insult": 1, "identity_hate": 0},
    ),
    (
        "What kind of bullcrap edits are these? Fuck off already.",
        {"toxic": 1, "severe_toxic": 0, "obscene": 1, "threat": 0, "insult": 0, "identity_hate": 0},
    ),
    (
        "You are a pathetic loser with nothing better to do than write bullshit articles.",
        {"toxic": 1, "severe_toxic": 0, "obscene": 1, "threat": 0, "insult": 1, "identity_hate": 0},
    ),
    # Severe abuse + Obscenity + Insult
    (
        "I hope you suffer and die in pain you worthless piece of human filth. Go rot.",
        {"toxic": 1, "severe_toxic": 1, "obscene": 1, "threat": 0, "insult": 1, "identity_hate": 0},
    ),
    (
        "You are the most vile, disgusting creature on earth. I will never stop destroying your work.",
        {"toxic": 1, "severe_toxic": 1, "obscene": 0, "threat": 0, "insult": 1, "identity_hate": 0},
    ),
    # Threatening language
    (
        "I know where you live and I will track you down and break your skull.",
        {"toxic": 1, "severe_toxic": 1, "obscene": 0, "threat": 1, "insult": 0, "identity_hate": 0},
    ),
    (
        "Watch your back next time you leave your house, you will pay for what you did.",
        {"toxic": 1, "severe_toxic": 0, "obscene": 0, "threat": 1, "insult": 0, "identity_hate": 0},
    ),
    (
        "I am going to put a bullet through your head if you touch this article again.",
        {"toxic": 1, "severe_toxic": 1, "obscene": 0, "threat": 1, "insult": 0, "identity_hate": 0},
    ),
    # Identity Hate
    (
        "All people from that country are filthy parasites and should be wiped out.",
        {"toxic": 1, "severe_toxic": 1, "obscene": 0, "threat": 0, "insult": 1, "identity_hate": 1},
    ),
    (
        "Your religion is a disease and followers like you are subhuman trash.",
        {"toxic": 1, "severe_toxic": 0, "obscene": 0, "threat": 0, "insult": 1, "identity_hate": 1},
    ),
    (
        "Go back to where you came from, your kind does not belong in civilized society.",
        {"toxic": 1, "severe_toxic": 0, "obscene": 0, "threat": 0, "insult": 1, "identity_hate": 1},
    ),
    # Obscenity only
    (
        "Damn this whole website is so damn slow and annoying today.",
        {"toxic": 0, "severe_toxic": 0, "obscene": 1, "threat": 0, "insult": 0, "identity_hate": 0},
    ),
]


def synthesize_record(
    index: int,
    is_toxic: bool,
    rng: random.Random,
) -> Dict[str, str | int]:
    """
    Constructs a single synthetic message record with appropriate multi-label flags.
    """
    record_hash = hashlib.md5(f"record-{index}-{rng.random()}".encode("utf-8")).hexdigest()[:16]

    if not is_toxic:
        base_message = rng.choice(CIVIL_MESSAGE_TEMPLATES)
        # Occasionally append civil continuation
        if rng.random() < 0.35:
            base_message += " " + rng.choice(CIVIL_MESSAGE_TEMPLATES)
        return {
            "id": record_hash,
            "comment_text": base_message,
            "toxic": 0,
            "severe_toxic": 0,
            "obscene": 0,
            "threat": 0,
            "insult": 0,
            "identity_hate": 0,
        }

    template, label_flags = rng.choice(TOXIC_MESSAGE_TEMPLATES)
    # Add slight random variations to text
    variations = [
        f"{template}",
        f"Hey, {template}",
        f"{template} Seriously.",
        f"Listen here: {template}",
    ]
    message_content = rng.choice(variations)

    return {
        "id": record_hash,
        "comment_text": message_content,
        "toxic": label_flags["toxic"],
        "severe_toxic": label_flags["severe_toxic"],
        "obscene": label_flags["obscene"],
        "threat": label_flags["threat"],
        "insult": label_flags["insult"],
        "identity_hate": label_flags["identity_hate"],
    }


def generate_benchmark_dataset(
    target_path: Path,
    sample_count: int = 1500,
    toxicity_ratio: float = 0.12,
    seed: int = 42,
) -> None:
    """
    Generates a synthetic CSV dataset matching Jigsaw schema with realistic class imbalance.
    """
    target_path.parent.mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)

    records: List[Dict[str, str | int]] = []
    toxic_target_count = int(sample_count * toxicity_ratio)
    civil_target_count = sample_count - toxic_target_count

    for index in range(civil_target_count):
        records.append(synthesize_record(index, is_toxic=False, rng=rng))

    for index in range(toxic_target_count):
        records.append(synthesize_record(index + civil_target_count, is_toxic=True, rng=rng))

    rng.shuffle(records)

    with open(target_path, mode="w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=CANONICAL_RAW_COLUMNS)
        writer.writeheader()
        writer.writerows(records)

    print(f"Generated {len(records)} benchmark records at: {target_path.resolve()}")
    print(f"Civil samples: {civil_target_count} ({civil_target_count/sample_count:.1%})")
    print(f"Toxic samples: {toxic_target_count} ({toxic_target_count/sample_count:.1%})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate synthetic benchmark dataset for CivilityAI.")
    parser.add_argument(
        "--count",
        type=int,
        default=1500,
        help="Number of records to synthesize (default: 1500)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="datasets/source/train.csv",
        help="Path where train.csv will be saved",
    )
    parser.add_argument(
        "--toxicity-ratio",
        type=float,
        default=0.14,
        help="Proportion of toxic records (default: 0.14)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility",
    )
    args = parser.parse_args()

    output_path = Path(args.output)
    generate_benchmark_dataset(
        target_path=output_path,
        sample_count=args.count,
        toxicity_ratio=args.toxicity_ratio,
        seed=args.seed,
    )
