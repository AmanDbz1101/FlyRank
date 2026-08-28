# System Design

## Overview

The Agentic Content Decline Investigator is a three-agent pipeline that investigates why content pages decline in search visibility. It takes a ranked queue of pages (from an ML model) and produces human-readable decline reports with evidence citations.

## Data Flow

```
Input: content page (or top-K from ranked queue)
  │
  ▼
┌──────────────────────────────────────────────────────────────────┐
│ STEP 1: FEATURE EXTRACTION                                       │
│ Warehouse.py queries DuckDB over HF parquets                     │
│ Produces: 8-feature vector + page metadata                       │
│ Decision moment: 2026-03-31                                      │
└──────────────────────┬───────────────────────────────────────────┘
                       │
                       ▼
┌──────────────────────────────────────────────────────────────────┐
│ STEP 2: PLANNER (rule-based)                                     │
│ Reads page signals, applies rules, generates 3-5 hypotheses     │
│ No LLM — deterministic, no token usage                           │
│ Output: list of (hypothesis_id, reason, priority)                │
└──────────────────────┬───────────────────────────────────────────┘
                       │
                       ▼
┌──────────────────────────────────────────────────────────────────┐
│ STEP 3: INVESTIGATOR (DuckDB queries)                            │
│ For each hypothesis:                                             │
│   - Builds a SQL query specific to the hypothesis                │
│   - Executes against the warehouse                               │
│   - Classifies evidence as SUPPORTS / CONTRADICTS / INCONCLUSIVE │
│ Output: list of (hypothesis_id, evidence_text, verdict)          │
└──────────────────────┬───────────────────────────────────────────┘
                       │
                       ▼
┌──────────────────────────────────────────────────────────────────┐
│ STEP 4: SYNTHESIZER (Groq LLaMA 3)                               │
│ Receives: page data + evidence list                              │
│ Generates: 3-5 sentence decline report with citations            │
│ Output: structured report with verdict + recommendation          │
└──────────────────────┬───────────────────────────────────────────┘
                       │
                       ▼
Output: decline investigation report (Markdown)
```

## Tech Stack

| Component | Technology | Why |
|-----------|-----------|-----|
| **Data warehouse** | DuckDB over HuggingFace parquets | Fast columnar queries, local caching, no download |
| **Feature vector** | pandas + numpy | Same 8 features as weekly assignments |
| **Planner** | Pure Python (if/else rules) | No LLM cost, deterministic, auditable |
| **Investigator** | DuckDB SQL queries | Reuses warehouse patterns from w03/w04 |
| **Synthesizer** | Groq API (LLaMA 3) | Free tier, fast, generates natural language |
| **Web app** | Streamlit | Interactive UI, easy deployment, Python-native |
| **Validation** | scikit-learn GroupKFold | Client-grouped, honest split design |

## Directory Structure

```
capstone/
├── src/
│   ├── agents/
│   │   ├── planner.py           # Rule-based hypothesis generator
│   │   ├── investigator.py      # DuckDB query executor
│   │   └── synthesizer.py       # Groq LLM report generator
│   ├── data/
│   │   ├── warehouse.py         # DuckDB + HF access
│   │   └── features.py          # Feature vector builder
│   ├── models/
│   │   ├── baseline.py          # Hand-crafted rule
│   │   └── classifier.py       # Logistic Regression
│   └── evaluation/
│       └── metrics.py           # Precision@K, leakage checks
├── app/
│   ├── streamlit_app.py         # Main entry point
│   ├── pages/
│   │   ├── 1_Research_Paper.py  # 9-section paper
│   │   ├── 2_Queue_Explorer.py  # Interactive queue browser
│   │   └── 3_Agent_Demo.py      # Run agent on pages
│   └── components/
│       ├── charts.py            # Plotly charts
│       └── agent_ui.py          # Agent output display
└── tests/
    ├── test_planner.py
    ├── test_investigator.py
    └── test_synthesizer.py
```

## External Dependencies

- **HuggingFace Hub**: Warehouse access (parquet files cached locally)
- **Groq API**: Synthesizer LLM (requires GROQ_API_KEY)
- **Streamlit Community Cloud**: Deployment (free tier)

## Security Model

- All data is pseudonymized (content_hash_id, client_hash_id)
- No client names, domains, URLs, or private queries
- GROQ_API_KEY stored in .env (never committed)
- HF_TOKEN stored in .env (never committed)
- Warehouse tokens are read-only
