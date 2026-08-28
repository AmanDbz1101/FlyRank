# Agentic Content Decline Investigator

A multi-agent system that autonomously investigates content decline — not just scores pages, but explains why each page is at risk.

## What This Is

This capstone project builds on the FlyRank ML Internship dataset (Lane 2: Refresh / Content Opportunity Scoring). It combines:

1. **A ranked action queue** — 125K content pages scored by decline risk
2. **An ML model** — Logistic Regression achieving P@50 = 0.88 (vs rule's 0.70)
3. **An agentic investigation system** — three agents that investigate *why* pages decline

The agents teach a core agentic AI skill: **searching for proper information** by decomposing questions into testable hypotheses, executing queries against a data warehouse, and synthesizing findings into human-readable reports.

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Streamlit Web App                         │
│  ┌──────────┐  ┌──────────────┐  ┌───────────────────────┐  │
│  │  Paper    │  │  Queue       │  │  Agent Demo           │  │
│  │  (9 sec)  │  │  Explorer    │  │  (run on pages)       │  │
│  └──────────┘  └──────────────┘  └───────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
                              │
                    ┌─────────┴─────────┐
                    │  Agent Pipeline    │
                    │                    │
                    │  ┌──────────────┐  │
                    │  │ Planner      │  │  Rule-based hypotheses
                    │  │ (rules)      │  │
                    │  └──────┬───────┘  │
                    │         │          │
                    │  ┌──────▼───────┐  │
                    │  │ Investigator │  │  DuckDB warehouse queries
                    │  │ (queries)    │  │
                    │  └──────┬───────┘  │
                    │         │          │
                    │  ┌──────▼───────┐  │
                    │  │ Synthesizer  │  │  Groq LLaMA 3 reports
                    │  │ (LLM)       │  │
                    │  └──────────────┘  │
                    └────────────────────┘
                              │
                    ┌─────────▼─────────┐
                    │  FlyRank Warehouse │
                    │  (DuckDB + HF)     │
                    │  125K pages        │
                    │  55 clients        │
                    └───────────────────┘
```

## Quick Start

```bash
cd capstone
pip install -r requirements.txt
streamlit run app/streamlit_app.py
```

Requires: Python 3.12+, HF_TOKEN in .env (for warehouse access), GROQ_API_KEY in .env (for Synthesizer).

## Project Structure

```
capstone/
├── README.md                     # This file
├── docs/                         # System documentation
│   ├── system-design.md          # Architecture, data flow
│   ├── agents.md                 # Agent specifications
│   ├── data-contract.md          # Tables, features, labels
│   ├── functional-requirements.md
│   ├── non-functional-requirements.md
│   ├── methodology.md            # ML approach, validation
│   └── limitations.md            # Honest framing
├── src/                          # Python implementation
│   ├── agents/                   # Planner, Investigator, Synthesizer
│   ├── data/                     # Warehouse access, feature builder
│   ├── models/                   # Baseline rule, ML model
│   └── evaluation/               # Metrics, leakage checks
├── app/                          # Streamlit web app
│   ├── streamlit_app.py          # Main entry point
│   ├── pages/                    # Paper, Explorer, Agent Demo
│   └── components/               # Reusable UI components
├── tests/                        # Agent tests
└── examples/                     # Sample reports
```

## Weekly Notebooks (completed)

| Notebook | Description | Status |
|----------|-------------|--------|
| `w01_research_question.ipynb` | Research question and lane | Done |
| `w02_ml_task_framing.ipynb` | ML task framing (ranking) | Done |
| `w03_data_contract.ipynb` | Data contract + features | Done |
| `w03_feature_leakage_check.ipynb` | Feature vector + leakage hunt | Done |
| `w04_baseline_score.ipynb` | Baseline rule (P@100 = 0.750) | Done |
| `w05_model.ipynb` | ML model (LR P@50 = 0.880) | Done |
| `w06_validation_audit.ipynb` | Validation audit | Pending |
| `w07_action_playbook.ipynb` | Action playbook | Pending |
| `capstone.ipynb` | Research paper notebook | Pending |

## Data

Built on the FlyRank ML Internship dataset — a pseudonymized, public-safe release of search intelligence data from real content sites.

- **Warehouse**: `FlyRank/internship-warehouse` on Hugging Face
- **Tables**: `fact_content_daily_performance`, `dim_content`, `dim_clients`
- **Population**: 125,573 visible content pages, 44 clients, March 2026 features, April 2026 labels

## Acknowledgments

Built on the FlyRank ML Internship dataset. Learn more at [flyrank.ai](https://flyrank.ai).
