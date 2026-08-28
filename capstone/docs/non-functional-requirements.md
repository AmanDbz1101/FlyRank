# Non-Functional Requirements

## Honesty Constraints

These are not suggestions — they are hard rules that override performance claims.

| ID | Constraint | Rationale |
|----|-----------|-----------|
| NF-01 | Claims MUST use the claim ladder: observed → directional → decision-support | Never causal without an experiment |
| NF-02 | MUST NOT claim "proves", "causes", "will increase", "the algorithm rewards" | Cross-sectional data doesn't support these |
| NF-03 | MUST NOT claim "predicted Google's algorithm" | We model outcomes in one portfolio, not Google's ranking |
| NF-04 | Base rate MUST appear next to every metric | Accuracy without base rate is misleading |
| NF-05 | Negative results ARE results — report them honestly | "We expected X; the data shows no such pattern" is valid |

## Privacy Constraints

| ID | Constraint | Rationale |
|----|-----------|-----------|
| NF-06 | MUST NOT include client names, domains, URLs, or private queries | Public-safe release |
| NF-07 | MUST NOT include raw query text or page URLs | Private data not shipped |
| NF-08 | MUST NOT include credentials or tokens in committed code | Security |
| NF-09 | MUST use only pseudonymous IDs (content_hash_id, client_hash_id) | Data protection |
| NF-10 | MUST NOT print or display raw warehouse rows with identifiable patterns | Anonymization |

## Leakage Constraints

| ID | Constraint | Rationale |
|----|-----------|-----------|
| NF-11 | Features MUST be knowable at the decision moment (2026-03-31) | No future data in features |
| NF-12 | MUST NOT use April/future-window columns as features | They contain the answer |
| NF-13 | MUST NOT use product flags (health_score, etc.) as features | Circular if rebuilt |
| NF-14 | MUST NOT use content_updated_date as a feature | Snapshot-as-of-build, 82.8% unknowable |
| NF-15 | Validation MUST use client-grouped splits | Random split overstates skill by ~0.025 AUC |

## Performance Constraints

| ID | Constraint | Rationale |
|----|-----------|-----------|
| NF-16 | Agent pipeline MUST complete in < 30 seconds per page | Interactive use |
| NF-17 | Streamlit app MUST load in < 5 seconds | User experience |
| NF-18 | Warehouse queries MUST complete in < 10 seconds each | Interactive use |
| NF-19 | Synthesizer MUST generate reports in < 10 seconds | Interactive use |
| NF-20 | Agent reports MUST be cached to avoid re-generation | Token budget (Groq free tier) |

## Deployment Constraints

| ID | Constraint | Rationale |
|----|-----------|-----------|
| NF-21 | MUST deploy to Streamlit Community Cloud (free tier) | No paid infrastructure |
| NF-22 | MUST work with HF_TOKEN + GROQ_API_KEY only | Minimal secrets |
| NF-23 | MUST NOT require GPU | Free tier doesn't have GPU |
| NF-24 | MUST NOT require downloading the full warehouse | 78M rows too large for Streamlit |
| NF-25 | MUST cache warehouse queries locally | Reduce API calls |

## Reproducibility Constraints

| ID | Constraint | Rationale |
|----|-----------|-----------|
| NF-26 | MUST fix random seeds (random_state=0) | Reproducible results |
| NF-27 | MUST commit metrics JSON after each run | Audit trail |
| NF-28 | MUST run notebooks top to bottom with no errors | Verification |
| NF-29 | MUST NOT commit datasets, .env, or large cache files | Repo hygiene |
| NF-30 | MUST include requirements.txt with pinned versions | Environment reproducibility |
