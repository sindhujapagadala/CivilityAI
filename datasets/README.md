# CivilityAI Dataset Directory

## Jigsaw Toxic Comment Classification Dataset

This directory holds the training and evaluation data for the CivilityAI content moderation system.

### Directory Structure

```text
datasets/
├── source/
│   └── train.csv          <-- Place the raw Kaggle Jigsaw train.csv here
├── prepared/
│   ├── sample_benchmark.csv
│   ├── validation_split.csv
│   └── evaluation_split.csv
└── sample_generator.py    <-- Generates realistic development and testing records
```

### Dataset Placement Instructions

1. Download `train.csv` from the Kaggle Competition:
   [Jigsaw Toxic Comment Classification Challenge](https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge/data)
2. Save the file directly as:
   `datasets/source/train.csv`
3. The expected columns in `train.csv` are:
   - `id`: Unique identifier for the comment
   - `comment_text`: Raw user-submitted comment string
   - `toxic`: Binary label (0 or 1)
   - `severe_toxic`: Binary label (0 or 1)
   - `obscene`: Binary label (0 or 1)
   - `threat`: Binary label (0 or 1)
   - `insult`: Binary label (0 or 1)
   - `identity_hate`: Binary label (0 or 1)

### Quick Start / Offline Testing

If you do not have the complete Kaggle dataset downloaded yet, generate a representative synthetic benchmark dataset using:

```bash
python datasets/sample_generator.py --count 1500 --output datasets/source/train.csv
```

This ensures the entire pipeline (EDA, baseline training, DistilBERT fine-tuning, threshold optimization, API, and tests) can be executed immediately out-of-the-box.
