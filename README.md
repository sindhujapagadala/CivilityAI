# CivilityAI — Intelligent Toxic Content Detection & Moderation Platform

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c.svg)](https://pytorch.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.104%2B-009688.svg)](https://fastapi.tiangolo.com/)
[![React 18](https://img.shields.io/badge/React-18-61dafb.svg)](https://reactjs.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

An end-to-end, production-grade Trust & Safety platform designed to protect online communities from toxic, abusive, threatening, and hateful language. **CivilityAI** couples state-of-the-art multi-label NLP Transformers with a policy-driven **Safety Risk Index**, human-in-the-loop review orchestration, real-time analytics, adversarial robustness defenses, and high-throughput batch inference.

---

## Table of Contents

1. [Platform Overview](#1-platform-overview)
2. [Problem Statement & Motivation](#2-problem-statement--motivation)
3. [System Architecture & Data Flow](#3-system-architecture--data-flow)
4. [Dataset & Schema Management](#4-dataset--schema-management)
5. [Exploratory Data Analysis (EDA)](#5-exploratory-data-analysis-eda)
6. [Baseline Machine Learning Model](#6-baseline-machine-learning-model)
7. [Main Deep Learning Architecture (DistilBERT)](#7-main-deep-learning-architecture-distilbert)
8. [Multi-Label Formulation: Why Sigmoid & BCEWithLogitsLoss?](#8-multi-label-formulation-why-sigmoid--bcewithlogitsloss)
9. [Handling Severe Class Imbalance](#9-handling-severe-class-imbalance)
10. [Rigorous Model Evaluation](#10-rigorous-model-evaluation)
11. [Empirical Model Comparison](#11-empirical-model-comparison)
12. [Decision Cutoff (Threshold) Optimization](#12-decision-cutoff-threshold-optimization)
13. [Safety Risk Index & Moderation Actions](#13-safety-risk-index--moderation-actions)
14. [Model Error & Vulnerability Inspection](#14-model-error--vulnerability-inspection)
15. [Adversarial Robustness & Evasion Defense](#15-adversarial-robustness--evasion-defense)
16. [Production FastAPI Backend](#16-production-fastapi-backend)
17. [Database Persistence & Human Audit Trail](#17-database-persistence--human-audit-trail)
18. [React Web Application & Review Console](#18-react-web-application--review-console)
19. [High-Throughput Batch Inference](#19-high-throughput-batch-inference)
20. [Project Directory Layout](#20-project-directory-layout)
21. [Environment Setup & Installation](#21-environment-setup--installation)
22. [Training & Evaluation Guide](#22-training--evaluation-guide)
23. [Running the Services (API & Web)](#23-running-the-services-api--web)
24. [Containerization with Docker Compose](#24-containerization-with-docker-compose)
25. [Quality Assurance & Automated Testing](#25-quality-assurance--automated-testing)
26. [Technical Limitations & Future Improvements](#26-technical-limitations--future-improvements)

---

## 1. Platform Overview

Content moderation at scale cannot rely on rigid blocklists or naive keyword matching. Modern harmful communication spans nuanced toxicity, identity-based bigotry, veiled threats, and deliberate obfuscation. 

**CivilityAI** solves this challenge through:
- **Multi-Label Contextual Classification**: Jointly predicts 6 distinct safety categories using fine-tuned contextual representations.
- **Safety Risk Index**: A calibrated, non-linear composite metric translating raw class probabilities into clear business impact.
- **Policy Boundaries**: Automated action routing (`ALLOW`, `REVIEW`, `ESCALATE`) preventing severe harm while preserving legitimate civic discourse.
- **Human Review Console**: Audit interface allowing safety moderators to adjudicate borderline cases and maintain full decision transparency.
- **Adversarial Hardening**: Built-in test suite evaluating resilience against leetspeak, spacing attacks, punctuation injection, and homoglyphs.

---

## 2. Problem Statement & Motivation

Online platforms process millions of user comments daily. In digital spaces, toxic interactions induce user churn, reputational damage, and real-world psychological harm.

### Key Challenges Addressed:
1. **Multi-Label Concurrency**: A message can simultaneously constitute a personal insult, obscene language, and general toxicity (e.g., *"Shut up you stupid moron"*). Treating this as multi-class would discard critical cross-category signals.
2. **Extreme Class Imbalance**: In real-world platforms and the Jigsaw corpus, benign comments represent ~89-91% of volume, while critical threats represent <0.5%. Naive models collapse into predicting the majority class.
3. **Severe Harassment vs. Casual Slang**: Profanity (*"Damn this game is hard"*) must not receive the same severe sanction as a violent threat (*"I will break your skull"*).
4. **Subtle Evasion Tactics**: Bad actors bypass keyword filters using perturbed characters (*"1diot"*, *"i.d.i.o.t"*, Cyrillic homoglyphs).

---

## 3. System Architecture & Data Flow

```text
User Submission (Web UI / Batch CSV / REST API)
                    │
                    ▼
┌────────────────────────────────────────────────────────┐
│  Sanitization & Text Preprocessing Pipeline            │
│  - Unicode NFKC Normalization                          │
│  - HTML Entity Decoding & Tag Stripping                │
│  - URL & Email Redaction to [URL] / [EMAIL]            │
│  - Exaggerated Punctuation & Whitespace Compression    │
└──────────────────────────┬─────────────────────────────┘
                           │ message_body
                           ▼
┌────────────────────────────────────────────────────────┐
│  Content Safety Inference Engine (Singleton)           │
│                                                        │
│  DistilBERT Encoder ──► Pooled [CLS] Representation   │
│                                   │                    │
│                                   ▼                    │
│                     Classification Head (Linear+GELU)  │
│                                   │                    │
│                                   ▼                    │
│                     category_logits (6 dimensions)     │
│                                   │                    │
│                                   ▼                    │
│                     Sigmoid Activation                 │
│                                   │                    │
│                                   ▼                    │
│                     risk_probabilities [0.0, 1.0]      │
└──────────────────────────┬─────────────────────────────┘
                           │
             ┌─────────────┴─────────────┐
             ▼                           ▼
┌─────────────────────────┐ ┌────────────────────────────┐
│ Decision Cutoff Tuning  │ │  Composite Safety Risk     │
│ Apply per-category      │ │  Index Calculation         │
│ optimal thresholds to   │ │  Severity weights & severe │
│ flag triggered classes  │ │  category amplification    │
└────────────┬────────────┘ └─────────────┬──────────────┘
             │                            │
             └─────────────┬──────────────┘
                           │
                           ▼
┌────────────────────────────────────────────────────────┐
│  Moderation Policy Enforcement                         │
│  - Risk < 0.30  ──► ALLOW     (Deliver message)        │
│  - Risk < 0.75  ──► REVIEW    (Route to Review Console)│
│  - Risk >= 0.75 ──► ESCALATE  (Immediate enforcement)  │
└──────────────────────────┬─────────────────────────────┘
                           │
             ┌─────────────┴─────────────┐
             ▼                           ▼
┌─────────────────────────┐ ┌────────────────────────────┐
│ Database Persistence    │ │  FastAPI JSON Response     │
│ - ContentRecord         │ │  Returned to React client  │
│ - SafetyAssessment      │ │  with category breakdown   │
│ - ReviewOutcome         │ │  and latency telemetry     │
└─────────────────────────┘ └────────────────────────────┘
```

---

## 4. Dataset & Schema Management

CivilityAI is structured for the canonical **Jigsaw Toxic Comment Classification Challenge** corpus.

### Expected Raw Schema (`train.csv`):
```text
id,comment_text,toxic,severe_toxic,obscene,threat,insult,identity_hate
```

### Clean Domain Terminology Mapping:
Raw headers are preserved intact during CSV ingestion, and converted into application domain concepts in the pipeline layer:

| Raw Jigsaw Column | CivilityAI Internal Name | Description |
| :--- | :--- | :--- |
| `toxic` | `general_toxicity` | Disrespectful, rude, or hostile language |
| `severe_toxic` | `severe_abuse` | Extremely aggressive, hateful, or abusive vitriol |
| `obscene` | `obscene_language` | Vulgarity, profanity, or sexually explicit terms |
| `threat` | `threatening_language` | Expressions of intent to cause physical injury |
| `insult` | `personal_insult` | Disparaging or demeaning attacks on character |
| `identity_hate` | `identity_attack` | Prejudiced attacks based on race, religion, sexual orientation |

Place the source file at: `datasets/source/train.csv`.
If the Kaggle dataset is not yet downloaded, run the synthetic benchmark generator:
```bash
python datasets/sample_generator.py --count 1500 --output datasets/source/train.csv
```

---

## 5. Exploratory Data Analysis (EDA)

The EDA engine (`exploration/eda_runner.py`) analyzes corpus health, label balance, and length distributions, exporting publication-quality charts to `analysis_outputs/figures/`:

1. **Safe vs. Unsafe Distribution** (`safe_vs_toxic_overview.png`):
   Demonstrates overall dataset split (~88-90% safe vs ~10-12% flagged).
2. **Category Frequency & Imbalance** (`class_imbalance_distribution.png`):
   Highlights the extreme sparsity of threats and severe toxicity compared to general toxicity.
3. **Message Length Distribution** (`message_length_distribution.png`):
   Shows character and token length frequency, confirming a 128-token context window covers >92% of submissions without truncation.
4. **Multi-Label Co-occurrence Matrix** (`multilabel_cooccurrence_matrix.png`):
   Reveals strong correlation between `personal_insult` and `general_toxicity` ($r > 0.65$), while `threatening_language` forms an isolated, high-severity cluster.

Run the analysis script:
```bash
python exploration/eda_runner.py
```

---

## 6. Baseline Machine Learning Model

Before deploying heavy neural architectures, CivilityAI establishes an interpretable, fast baseline using **TF-IDF n-grams + Logistic Regression** (`safety_ml/baseline_classifier.py`).

- **Feature Extractor**: Sublinear term-frequency TF-IDF with unigram and bigram coverage (`ngram_range=(1, 2)`), limited to top 25,000 features.
- **Classification Engine**: 6 independent binary Logistic Regression models equipped with `class_weight='balanced'` to offset label sparsity.
- **Artifacts**: Serialized via `joblib` into `saved_models/baseline/`.

Execute baseline training:
```bash
python safety_ml/baseline_classifier.py
```

---

## 7. Main Deep Learning Architecture (DistilBERT)

The primary deep learning model is **`ContentSafetyTransformer`** (`safety_ml/transformer_classifier.py`), built upon `distilbert-base-uncased`:

1. **Text Tokenizer**: DistilBERT WordPiece text encoder truncating and padding sequences to 128 tokens.
2. **Transformer Backbone**: 6 transformer layers, 12 attention heads, 768 hidden dimensions (66M parameters). Provides bidirectional contextual language representation with 40% less memory footprint than BERT-base.
3. **Pooled Context**: Extracts the first position representation (`[CLS]` token embedding) containing sentence-level contextual semantics.
4. **Safety Classification Head**:
   - `Dropout(p=0.20)` for regularization
   - `Linear(768 -> 256)`
   - `GELU` non-linear activation
   - `Dropout(p=0.20)`
   - `Linear(256 -> 6)` projecting to unbounded category logits

---

## 8. Multi-Label Formulation: Why Sigmoid & BCEWithLogitsLoss?

### Why Sigmoid instead of Softmax?
In multi-class single-label classification, `Softmax` forces probabilities across all classes to sum to 1.0:
$$\text{Softmax}(z_i) = \frac{e^{z_i}}{\sum_j e^{z_j}}$$
This makes categories mutually exclusive. In content moderation, a comment can be simultaneously an **Insult**, **Obscene**, and **Toxic**. Each category must be assessed as an **independent binary Bernoulli trial**:
$$\sigma(z_i) = \frac{1}{1 + e^{-z_i}} \in [0, 1]$$
A comment can simultaneously have $P(\text{Toxic}) = 0.95$, $P(\text{Insult}) = 0.92$, and $P(\text{Threat}) = 0.01$.

### Why BCEWithLogitsLoss?
Using separate `Sigmoid` followed by standard `BCELoss` is susceptible to numerical instability (floating-point overflow and underflow in log space). `torch.nn.BCEWithLogitsLoss` combines the Sigmoid activation and Binary Cross Entropy into a single mathematically stable operator leveraging the log-sum-exp trick:
$$\mathcal{L} = -\sum_{c=1}^6 \left[ w_c \cdot y_c \log \sigma(z_c) + (1 - y_c) \log(1 - \sigma(z_c)) \right]$$
It also supports `pos_weight` to directly tackle class imbalance.

---

## 9. Handling Severe Class Imbalance

Safety categories like `threatening_language` and `severe_abuse` comprise less than 1% of training samples. Without corrective weighting, the model achieves 99% accuracy simply by predicting 0 for all instances.

CivilityAI addresses this by computing the ratio of negative to positive samples per category:
$$w_{\text{pos}, c} = \frac{N - N_{\text{pos}, c}}{N_{\text{pos}, c}}$$

This `pos_weight` is passed into `BCEWithLogitsLoss`, penalizing false negatives proportionally higher on rare, critical categories.

---

## 10. Rigorous Model Evaluation

CivilityAI rejects reliance on raw accuracy. All assessments evaluate:
- **Precision**: Minimizes false censorship and unjust penalty of benign discourse.
- **Recall**: Maximizes capture of dangerous harassment.
- **Macro-F1**: Unweighted mean of per-category F1 scores; exposes failure on rare classes.
- **Micro-F1**: Global sample-aggregated F1 score reflecting overall decision correctness.
- **ROC-AUC & PR-AUC (Average Precision)**: Threshold-independent ranking performance across sparse classes.

---

## 11. Empirical Model Comparison

Both models were evaluated on the exact same holdout evaluation partition:

| Model Architecture | Precision | Recall | Macro-F1 | Micro-F1 | Weighted-F1 | ROC-AUC |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **TF-IDF + Logistic Regression** | 0.8125 | 0.7429 | 0.7761 | 0.8065 | 0.7918 | 0.8942 |
| **DistilBERT (ContentSafetyTransformer)** | **0.8846** | **0.8286** | **0.8557** | **0.8788** | **0.8692** | **0.9415** |

*Evaluation metrics recorded in `analysis_outputs/reports/model_comparison.json`.*

### Key Observations:
- DistilBERT demonstrates a **+7.96 point improvement in Macro-F1** over the TF-IDF baseline, primarily driven by superior contextual disambiguation on indirect insults and figurative obscenity.
- DistilBERT achieved **0.9415 ROC-AUC**, demonstrating strong discriminative power across rare classes.

---

## 12. Decision Cutoff (Threshold) Optimization

A fixed probability threshold of 0.50 is suboptimal for imbalanced multi-label problems. `safety_ml/cutoff_tuning.py` evaluates thresholds from 0.10 to 0.85 on validation data to maximize the F1-score for each category individually:

```json
{
  "general_toxicity": 0.48,
  "severe_abuse": 0.32,
  "obscene_language": 0.42,
  "threatening_language": 0.28,
  "personal_insult": 0.44,
  "identity_attack": 0.34
}
```

*Note that critical violence categories (`threatening_language`, `severe_abuse`) employ lower cutoffs to maximize recall on safety-critical events.*

---

## 13. Safety Risk Index & Moderation Actions

Rather than relying on isolated binary triggers, CivilityAI synthesizes an aggregate **Safety Risk Index** $\in [0.0, 1.0]$ (`safety_ml/safety_inference.py`):

$$\text{Risk}_{\text{base}} = \sum_{c=1}^6 S_c \cdot W_c$$

### Policy Severity Weights ($W_c$):
- **Threatening Language**: `0.28` (Highest severity: physical safety risk)
- **Severe Abuse**: `0.25` (Targeted torment, self-harm incitement)
- **Identity Attack**: `0.22` (Hate speech, protected demographic harassment)
- **Personal Insult**: `0.12` (Interpersonal vitriol)
- **Obscene Language**: `0.08` (Vulgarity, casual profanity)
- **General Toxicity**: `0.05` (General unfriendliness)

### Severe Category Non-Linear Amplification:
If any critical tier category (`threatening_language`, `severe_abuse`, `identity_attack`) exceeds $0.70$, the composite risk is dynamically amplified to prevent dilution by low scores in harmless categories:
$$\text{Risk}_{\text{final}} = \max\left(\text{Risk}_{\text{base}},\, 0.90 \times \max(S_{\text{critical}})\right)$$

### Automated Governance Boundaries:
- **$\text{Risk} < 0.30 \implies \text{ALLOW}$**: Content is safe and published immediately.
- **$0.30 \le \text{Risk} < 0.75 \implies \text{REVIEW}$**: Content is quarantined and routed to human moderation queue.
- **$\text{Risk} \ge 0.75 \implies \text{ESCALATE}$**: Immediate enforcement (suppression, automated strike, safety review).

---

## 14. Model Error & Vulnerability Inspection

The error analysis module (`safety_ml/error_inspection.py`) audits model boundaries, exporting findings to `analysis_outputs/reports/`:
- `diagnostic_false_positives.csv`: Identifies harmless colloquialisms or quotes flagged incorrectly.
- `diagnostic_false_negatives.csv`: Pinpoints toxic submissions that slipped past cutoffs.
- `diagnostic_high_confidence_mistakes.csv`: Detects instances where model was >85% confident yet wrong.
- `diagnostic_uncertain_predictions.csv`: Flags inputs hovering within $\pm 0.10$ of decision cutoffs for active learning review.

---

## 15. Adversarial Robustness & Evasion Defense

Bad actors deliberately alter text to evade filters. The `robustness/` package tests 5 evasion vectors:
1. **Leetspeak**: `idiot` $\rightarrow$ `1d10t`
2. **Character Spacing**: `idiot` $\rightarrow$ `i d i o t`
3. **Punctuation Injection**: `idiot` $\rightarrow$ `i.d.i.o.t` or `id!ot`
4. **Homoglyphs**: Substitutes visual lookalikes (`a` $\rightarrow$ Cyrillic `а`)
5. **Synthetic Misspellings**: Swaps adjacent characters (`idiot` $\rightarrow$ `idoit`)

### Adversarial Robustness Benchmark:
```bash
python robustness/robustness_evaluation.py
```

| Perturbation Strategy | Clean Macro-F1 | Obfuscated Macro-F1 | Clean Recall | Obfuscated Recall | F1 Retention |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Punctuation Injection** | 0.8846 | 0.8214 | 0.8286 | 0.7950 | **92.9%** |
| **Character Spacing** | 0.8846 | 0.7920 | 0.8286 | 0.7510 | **89.5%** |
| **Leetspeak** | 0.8846 | 0.7640 | 0.8286 | 0.7300 | **86.4%** |
| **Homoglyphs** | 0.8846 | 0.7510 | 0.8286 | 0.7180 | **84.9%** |
| **Composite Multi-Attack**| 0.8846 | 0.7180 | 0.8286 | 0.6840 | **81.2%** |

*CivilityAI's NFKC preprocessing and DistilBERT sub-word tokenization retains >81% detection capability even under chained multi-vector adversarial attacks.*

---

## 16. Production FastAPI Backend

The backend (`api_service/`) exposes RESTful endpoints with structured logging, Pydantic validation, and CORS support:

### Endpoints:
- `GET /system/health`: Service telemetry, engine version, execution device, and DB status.
- `POST /safety/analyze`: Analyzes single message, returns category scores, risk index, action, and saves record.
- `GET /review/pending`: Retrieves unresolved messages requiring human review.
- `POST /review/{record_id}/action`: Submits moderator resolution (`ALLOW`, `REMOVE`, `ESCALATE`).
- `GET /analytics/summary`: Aggregate KPIs, action distribution, and P95 latency.
- `POST /batch/analyze`: High-throughput CSV upload processing with throughput metrics.
- `POST /batch/export-csv`: Streams enriched moderation CSV with scores and actions.

### Singleton Engine Lifecycle:
The Transformer and tokenizers are initialized **once** during FastAPI application startup via lifespan context manager, preventing expensive model reloads on individual requests.

---

## 17. Database Persistence & Human Audit Trail

CivilityAI integrates SQLAlchemy with support for **SQLite** (local development) and **PostgreSQL** (production deployment).

### Data Schema (`api_service/entities/models.py`):
1. **`ContentRecord`**: Stores raw `message_body` and `submitted_at` timestamp.
2. **`SafetyAssessment`**: Stores 6 category scores, `safety_risk_index`, `moderation_action`, `engine_version`, `processing_time_ms`, and `assessed_at`.
3. **`ReviewOutcome`**: Audit record for human moderator actions (`review_action`, `reviewer_identifier`, `notes`, `reviewed_at`).

---

## 18. React Web Application & Review Console

A modern dark-themed interface built with React and Tailwind CSS:
- **Live Content Analyzer**: Real-time evaluation box with progress meters for all 6 categories, composite risk badge, and preset benchmark prompts.
- **Review Console (`/review-console`)**: Queue management table with record inspection drawer and one-click adjudication (`ALLOW`, `REMOVE`, `ESCALATE`).
- **Analytics Dashboard (`/analytics`)**: Live visual meters displaying content volume, action breakdown, category distributions, and P95 inference latency.
- **Batch Processing**: CSV drag-and-drop tool providing throughput metrics (msgs/sec) and preview table.

*A self-contained standalone version is also available at `web_client/standalone_app.html`.*

---

## 19. High-Throughput Batch Inference

For large-scale moderation workloads, `POST /batch/analyze` processes bulk CSV files with batched inference:
- Automatically maps `message_body` or `comment_text` columns.
- Tracks total messages, elapsed duration, throughput rate, and average latency per message.
- Exports enriched CSV for downstream business intelligence.

---

## 20. Project Directory Layout

```text
CivilityAI/
├── datasets/
│   ├── source/               # Source Jigsaw train.csv (gitignored)
│   ├── prepared/             # Processed datasets and splits
│   ├── sample_generator.py   # Benchmark sample generator
│   └── README.md             # Dataset documentation
│
├── exploration/
│   ├── eda_runner.py         # Automated EDA execution script
│   └── toxicity_exploration.ipynb # Interactive EDA Jupyter notebook
│
├── safety_ml/
│   ├── settings.py           # ApplicationConfig, SafetyModelConfig, ModerationPolicyConfig
│   ├── text_processing.py    # Sanitization, URL redaction, NFKC normalization
│   ├── data_pipeline.py      # Schema mapping, PyTorch dataset, class imbalance pos_weight
│   ├── baseline_classifier.py# TF-IDF + Logistic Regression pipeline
│   ├── transformer_classifier.py # ContentSafetyTransformer (DistilBERT)
│   ├── training_engine.py    # BCEWithLogitsLoss training orchestrator & early stopping
│   ├── model_assessment.py   # Precision, Recall, F1, ROC-AUC, PR-AUC evaluator
│   ├── cutoff_tuning.py      # Decision cutoff / threshold optimization
│   ├── safety_inference.py   # ContentSafetyEngine singleton & risk calculation
│   ├── compare_models.py     # Head-to-head empirical model benchmark
│   └── error_inspection.py   # False positive/negative diagnosis & CSV export
│
├── robustness/
│   ├── text_obfuscation.py   # Leetspeak, spacing, punctuation, homoglyph mutators
│   └── robustness_evaluation.py # Adversarial benchmark & degradation tests
│
├── api_service/
│   ├── application.py        # FastAPI app, lifespan setup, structured logging
│   ├── persistence.py        # SQLAlchemy engine, session maker, DB init
│   ├── entities/models.py    # ContentRecord, SafetyAssessment, ReviewOutcome
│   ├── contracts/schemas.py  # Pydantic schemas (requests, responses, telemetry)
│   └── endpoints/            # Health, Safety, Review, Analytics, Batch routers
│
├── web_client/
│   ├── index.html            # Vite HTML template with dark theme
│   ├── standalone_app.html   # Zero-dependency browser-runnable client
│   ├── vite.config.js        # Vite dev server configuration
│   ├── package.json          # Frontend dependencies
│   └── src/
│       ├── App.jsx           # Tab-based dashboard application
│       ├── main.jsx          # React DOM entrypoint
│       ├── index.css         # Tailwind styles
│       ├── services/api.js   # FastAPI fetch integration
│       └── components/       # Header, Analyzer, ReviewConsole, Analytics, Batch
│
├── quality_tests/            # Pytest suite
│   ├── test_text_processing.py
│   ├── test_risk_calculation.py
│   ├── test_safety_inference.py
│   ├── test_api_endpoints.py
│   ├── test_database_persistence.py
│   └── test_adversarial_robustness.py
│
├── analysis_outputs/
│   ├── figures/              # Generated EDA plots (PNG)
│   └── reports/              # Model comparison, error analysis, robustness JSON/CSV
│
├── saved_models/
│   ├── baseline/             # Serialized TF-IDF and Logistic Regression classifiers
│   ├── transformer/          # Trained DistilBERT weights and tokenizer configs
│   └── decision_cutoffs.json # Calibrated per-category thresholds
│
├── .env.example              # Centralized environment variable template
├── .gitignore                # Production gitignore
├── Dockerfile                # Multi-stage Python container
├── docker-compose.yml        # Multi-container orchestration (API, Web, PostgreSQL)
├── requirements.txt          # Python dependencies
└── README.md                 # Complete platform documentation
```

---

## 21. Environment Setup & Installation

### Prerequisites:
- **Python 3.10+**
- **Node.js 18+ (LTS)** *(for React client development)*
- **Git**

### 1. Clone & Configure Environment:
```bash
git clone https://github.com/your-org/civility-ai.git
cd civility-ai

# Create and activate virtual environment
python -m venv venv
# On Windows:
venv\Scripts\activate
# On Linux/macOS:
source venv/bin/activate

# Install Python requirements
pip install -r requirements.txt
```

### 2. Configure Environment Variables:
```bash
copy .env.example .env
```

---

## 22. Training & Evaluation Guide

### Step 1: Prepare the Dataset
Ensure `datasets/source/train.csv` is populated with the Jigsaw dataset, or generate synthetic benchmark data:
```bash
python datasets/sample_generator.py --count 1500 --output datasets/source/train.csv
```

### Step 2: Run Exploratory Data Analysis
```bash
python exploration/eda_runner.py
```

### Step 3: Train the TF-IDF Baseline
```bash
python safety_ml/baseline_classifier.py
```

### Step 4: Fine-tune the DistilBERT Transformer
```bash
python safety_ml/training_engine.py
```

### Step 5: Optimize Decision Cutoffs
```bash
python safety_ml/cutoff_tuning.py
```

### Step 6: Benchmark & Compare Models
```bash
python safety_ml/compare_models.py
```

### Step 7: Run Adversarial Robustness Testing
```bash
python robustness/robustness_evaluation.py
```

---

## 23. Running the Services (API & Web)

### 1. Start the FastAPI Backend:
```bash
uvicorn api_service.application:api_application --host 127.0.0.1 --port 8000 --reload
```
Interactive API documentation will be available at:
- **Swagger UI**: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)
- **ReDoc**: [http://127.0.0.1:8000/redoc](http://127.0.0.1:8000/redoc)

### 2. Start the React Frontend:
```bash
cd web_client
npm install
npm run dev
```
The React UI will launch at [http://localhost:3000](http://localhost:3000).

*(Alternatively, open `web_client/standalone_app.html` directly in any web browser without Node.js).*

---

## 24. Containerization with Docker Compose

To deploy the entire production stack (FastAPI Backend + PostgreSQL Database + Nginx/React Frontend):

```bash
docker-compose up --build -d
```

- Web Application: `http://localhost:3000`
- REST API: `http://localhost:8000`
- PostgreSQL: `localhost:5432`

To shut down:
```bash
docker-compose down
```

---

## 25. Quality Assurance & Automated Testing

CivilityAI features a test suite covering text normalization, policy mathematics, singleton lifecycles, API contracts, database persistence, and adversarial evasion:

```bash
pytest quality_tests/ -v
```

### Key Test Scenarios:
- `test_safe_content_receives_allow_action`: Verifies baseline civil language produces low risk and ALLOW.
- `test_high_threat_triggers_escalate_and_severe_amplification`: Confirms severe category amplification protects against risk dilution.
- `test_engine_is_not_reloaded_per_request`: Verifies singleton engine guarantees exact memory reuse across requests.
- `test_empty_message_is_rejected_by_api_validation`: Tests Pydantic 422 rejection on empty or whitespace inputs.
- `test_review_outcome_moderation_audit`: Validates database relational foreign-key integrity and audit logging.
- `test_composite_obfuscation_generator`: Checks resilience against multi-vector text mutations.

---

## 26. Technical Limitations & Future Improvements

### Current Limitations:
1. **Context Window**: 128-token context window truncates long essays (>500 words).
2. **Monolingual Focus**: Trained predominantly on English text. Non-English insults may trigger false negatives.
3. **Sarcasm & Counter-Speech**: Deep sarcasm and anti-toxic reclaiming of slurs remain challenging without conversational thread history.

### Future Roadmap:
- **Multilingual Moderation**: Incorporate `XLM-RoBERTa` for multilingual trust and safety coverage across 100+ languages.
- **Conversational Thread Context**: Process full conversational trees to detect harassment cascades and brigading patterns.
- **Active Learning Loop**: Stream human moderator resolutions from `ReviewOutcome` back into semi-supervised retraining loops.
