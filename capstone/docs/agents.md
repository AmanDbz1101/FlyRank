# Agent Specifications

## Overview

The system has three agents, each with a distinct role:

| Agent | Role | Input | Output | LLM? |
|-------|------|-------|--------|------|
| **Planner** | Generate hypotheses | Page signals | List of hypotheses | No (rules) |
| **Investigator** | Test hypotheses | Hypotheses + page data | Evidence per hypothesis | No (SQL) |
| **Synthesizer** | Generate report | Page data + evidence | Structured report | Yes (Groq) |

---

## Agent 1: Planner (Rule-Based)

### Role
Decompose a page's signals into testable hypotheses about why it might be declining.

### Input
A dictionary of page signals:
```python
{
    "content_hash_id": "content_xxx",
    "content_type": "keyword article",
    "mar_impressions": 23918,
    "ctr_mar": 0.531,
    "mar_avg_position": 4.649,
    "momentum_feb_to_mar_pct": -4.466,
    "has_clicks": 1,
    "has_feb_data": 1,
    "has_ga4": 0,
    "baseline_score": 3.5,
    "rf_prob": 0.45,
    "lr_prob": 0.62,
}
```

### Rules

| # | Condition | Hypothesis ID | Reason | Priority |
|---|-----------|---------------|--------|----------|
| 1 | `ctr_mar < 0.25 AND mar_avg_position ≤ 10` | `SNIPPET_GAP` | Page ranks well but earns few clicks — snippet may underperform its rank | 1 (high) |
| 2 | `momentum > 25` | `SPIKE_REVERSION` | March was a traffic spike; may revert to prior level | 2 |
| 3 | `momentum < -25` | `ACTIVE_DECLINE` | Page is actively losing traffic month-over-month | 1 (high) |
| 4 | `has_ga4 == 0` | `UNKNOWN_ENGAGEMENT` | No GA4 data — engagement cannot be assessed | 3 (low) |
| 5 | `ctr_mar > 1.5 × ctr_expected` | `EXTERNAL_FACTOR` | CTR is healthy; decline may be caused by external factors (competition, SERP changes) | 2 |
| 6 | `content_age_days > 365` (from dim_content) | `CONTENT_STALE` | Page hasn't been updated in over a year | 2 |
| 7 | Always (fallback) | `GENERAL_RISK` | Page shows general decline risk signals | 3 (low) |

Rules fire in priority order. A page can generate 1–5 hypotheses (not all rules fire).

### Output
```python
[
    {"id": "SNIPPET_GAP", "reason": "CTR 0.531% at position 4.6 — below expected for this band", "priority": 1},
    {"id": "ACTIVE_DECLINE", "reason": "Momentum -4.5% — page is losing traffic", "priority": 1},
    {"id": "UNKNOWN_ENGAGEMENT", "reason": "No GA4 data available for this client", "priority": 3},
]
```

### Why Rule-Based (Not LLM)
- **No token cost**: Planner runs on every page; LLM would burn tokens fast
- **Deterministic**: Same inputs → same hypotheses → auditable
- **Fast**: Rule evaluation is O(1) per page
- **The LLM's job is explanation, not hypothesis generation**

---

## Agent 2: Investigator (DuckDB Queries)

### Role
Test each hypothesis by executing warehouse queries and classifying the evidence.

### Input
- List of hypotheses from Planner
- Page signals (for query parameters)

### Query Patterns

| Hypothesis | SQL Pattern | Evidence Classification |
|-----------|-------------|------------------------|
| `SNIPPET_GAP` | Compare page CTR to position-band mean CTR | SUPPORTS if CTR < 0.5× band mean; CONTRADICTS if > 1.0× |
| `SPIKE_REVERSION` | Check if Feb impressions were anomalously low | SUPPORTS if Feb < 50% of Mar; CONTRADICTS if Feb ≈ Mar |
| `ACTIVE_DECLINE` | Verify April < 80% of March (the label) | SUPPORTS if True; CONTRADICTS if False |
| `UNKNOWN_ENGAGEMENT` | Check GA4 availability flag | INCONCLUSIVE (cannot test) |
| `EXTERNAL_FACTOR` | Compare page CTR to client-wide CTR trend | SUPPORTS if page declined faster than client average |
| `CONTENT_STALE` | Check content_created_date vs decision moment | SUPPORTS if > 365 days; CONTRADICTS if < 180 days |
| `GENERAL_RISK` | Compute all features, return summary | INCONCLUSIVE (descriptive only) |

### Output
```python
[
    {"hypothesis": "SNIPPET_GAP", "evidence": "CTR 0.531% vs 0.34% expected — better than average", "verdict": "CONTRADICTS"},
    {"hypothesis": "ACTIVE_DECLINE", "evidence": "Momentum -4.5%, confirmed by April data", "verdict": "SUPPORTS"},
    {"hypothesis": "UNKNOWN_ENGAGEMENT", "evidence": "No GA4 data for client", "verdict": "INCONCLUSIVE"},
]
```

### Evidence Verdicts

| Verdict | Meaning |
|---------|---------|
| `SUPPORTS` | Evidence confirms the hypothesis |
| `CONTRADICTS` | Evidence refutes the hypothesis |
| `INCONCLUSIVE` | Cannot determine (insufficient data, or query returned no results) |

---

## Agent 3: Synthesizer (Groq LLaMA 3)

### Role
Combine page data + evidence into a structured, human-readable decline report.

### Input
- Page signals (CTR, position, momentum, volume, content type)
- Evidence list from Investigator

### Prompt Template
```
You are a content decline investigator for an SEO content team. Given the following
page data and evidence from your investigation, write a 3-5 sentence decline report.

Page data:
- Content type: {content_type}
- March impressions: {mar_impressions}
- March CTR: {ctr_mar}% (expected: {ctr_expected}%)
- Average position: {mar_avg_position}
- Feb→Mar momentum: {momentum}%
- Has GA4 data: {has_ga4}

Evidence from investigation:
{evidence_list}

Write the report in this format:
1. One sentence: what happened (the decline pattern)
2. 1-2 sentences: evidence supporting or contradicting decline
3. One sentence: what the model/agent thinks is the primary driver
4. One sentence: recommended action

Be specific. Cite numbers. Do not make causal claims.
```

### Output
```markdown
## Decline Investigation Report
**Page:** content_xxx | **Type:** keyword article | **March impressions:** 23,918

### Verdict: ACTIVE DECLINE (moderate confidence)

The page's search impressions declined ~4.5% from February to March, placing it
in the declining category (April < 80% of March threshold). Despite a healthy CTR
of 0.53% (better than the 0.34% expected for its position band), the page's traffic
trajectory points to ongoing decline.

The primary driver appears to be a momentum shift — the page may have benefited
from a temporary surge in February that is now reverting. No GA4 engagement data
is available to confirm whether user behavior changed.

### Recommendation: MONITOR_30D
Re-check in 30 days. If impressions continue to fall, consider a content refresh.
```

### Configuration
- **Model**: `llama3-8b-8192` (Groq free tier)
- **Max tokens**: 300
- **Temperature**: 0.3 (conservative, factual)
- **Rate limit**: Groq free tier limits apply; cache reports to avoid re-generation

---

## Agent Interaction Diagram

```
                    ┌─────────────────┐
                    │  Page (input)   │
                    └────────┬────────┘
                             │
                    ┌────────▼────────┐
                    │   Planner       │
                    │   (rules)       │
                    └────────┬────────┘
                             │ hypotheses
                    ┌────────▼────────┐
                    │  Investigator   │
                    │  (DuckDB)       │
                    └────────┬────────┘
                             │ evidence
                    ┌────────▼────────┐
                    │  Synthesizer    │
                    │  (Groq LLM)     │
                    └────────┬────────┘
                             │
                    ┌────────▼────────┐
                    │  Report (output)│
                    └─────────────────┘
```

## Error Handling

| Agent | Failure Mode | Recovery |
|-------|-------------|----------|
| Planner | No rules fire | Generate `GENERAL_RISK` as fallback |
| Investigator | Query returns empty | Mark hypothesis as `INCONCLUSIVE` |
| Investigator | DuckDB error | Log error, skip hypothesis, continue |
| Synthesizer | Groq API error | Return evidence summary as plain text |
| Synthesizer | Token limit exceeded | Truncate evidence, retry |
| Synthesizer | Rate limit hit | Wait and retry, or return cached report |

## Testing Strategy

- **Planner**: Unit tests for each rule, edge cases (NaN values, missing data)
- **Investigator**: Integration tests with cached warehouse data
- **Synthesizer**: Smoke test with sample inputs, verify output format
- **End-to-end**: Run on 10-20 sample pages, review reports manually
