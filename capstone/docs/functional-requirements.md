# Functional Requirements

## System-Level Requirements

| ID | Requirement | Priority |
|----|-------------|----------|
| FR-01 | The system MUST rank 125K+ content pages by decline risk | High |
| FR-02 | The system MUST generate a decline investigation report for any given page | High |
| FR-03 | The system MUST use only signals knowable at the decision moment (2026-03-31) | High |
| FR-04 | The system MUST not use any April/future-window data as features | High |
| FR-05 | The system MUST not use product flags (health_score, etc.) as features | High |
| FR-06 | The system MUST present all 9 sections of the research paper | High |
| FR-07 | The system MUST allow users to explore the ranked queue interactively | Medium |
| FR-08 | The system MUST allow users to run the agent on specific pages | Medium |
| FR-09 | The system MUST cache agent reports to avoid re-generation | Medium |
| FR-10 | The system MUST work with only HF_TOKEN and GROQ_API_KEY | High |

## Agent-Specific Requirements

### Planner

| ID | Requirement | Priority |
|----|-------------|----------|
| PL-01 | MUST generate at least 1 hypothesis per page | High |
| PL-02 | MUST generate no more than 5 hypotheses per page | Medium |
| PL-03 | MUST be deterministic (same inputs → same hypotheses) | High |
| PL-04 | MUST NOT use any LLM (rule-based only) | High |
| PL-05 | MUST assign a priority to each hypothesis | Medium |
| PL-06 | MUST include a human-readable reason for each hypothesis | High |

### Investigator

| ID | Requirement | Priority |
|----|-------------|----------|
| IN-01 | MUST execute at least one query per hypothesis | High |
| IN-02 | MUST classify each piece of evidence as SUPPORTS/CONTRADICTS/INCONCLUSIVE | High |
| IN-03 | MUST use only warehouse data (DuckDB over HF parquets) | High |
| IN-04 | MUST NOT query April/future-window data | High |
| IN-05 | MUST handle query errors gracefully (skip, log, continue) | Medium |
| IN-06 | MUST include row counts with every evidence statement | Medium |

### Synthesizer

| ID | Requirement | Priority |
|----|-------------|----------|
| SY-01 | MUST generate a 3-5 sentence report per page | High |
| SY-02 | MUST cite specific evidence from the Investigator | High |
| SY-03 | MUST NOT make causal claims | High |
| SY-04 | MUST recommend an action (REFRESH/MONITOR/NO_ACTION) | High |
| SY-05 | MUST use Groq LLaMA 3 (or equivalent free-tier LLM) | High |
| SY-06 | MUST cache reports to avoid re-generation | Medium |
| SY-07 | MUST handle API errors gracefully (return evidence as plain text) | Medium |

## Streamlit App Requirements

| ID | Requirement | Priority |
|----|-------------|----------|
| AP-01 | MUST have a Research Paper tab with all 9 sections | High |
| AP-02 | MUST have a Queue Explorer tab with sortable/filterable table | Medium |
| AP-03 | MUST have an Agent Demo tab that runs the agent pipeline | High |
| AP-04 | MUST display charts (precision@K, feature importance, decile analysis) | Medium |
| AP-05 | MUST deploy to Streamlit Community Cloud | High |
| AP-06 | MUST load .env automatically for local development | Medium |
| AP-07 | MUST NOT expose API keys in the UI | High |

## Evaluation Requirements

| ID | Requirement | Priority |
|----|-------------|----------|
| EV-01 | MUST compare model vs baseline on the same split and metric | High |
| EV-02 | MUST use client-grouped validation (GroupKFold, 4 folds) | High |
| EV-03 | MUST report precision@K at K = 10, 50, 100, 500 | High |
| EV-04 | MUST report ROC AUC | High |
| EV-05 | MUST print base rate next to every metric | High |
| EV-06 | MUST include error analysis (by content type, momentum band) | High |
| EV-07 | MUST include feature importances (RF) and coefficients (LR) | Medium |
| EV-08 | MUST test agent on 10-20 sample pages | Medium |

## Reproducibility Requirements

| ID | Requirement | Priority |
|----|-------------|----------|
| RP-01 | MUST fix random seeds (random_state=0) | High |
| RP-02 | MUST note library versions if headline number matters | Medium |
| RP-03 | MUST commit metrics JSON to work/outputs/ | High |
| RP-04 | MUST run notebooks top to bottom with no errors | High |
| RP-05 | MUST NOT commit datasets or .env files | High |
