# Methodology

## ML Approach

### Problem Framing

**Lane**: Refresh / Content Opportunity Scoring (Lane 2)

**Task**: Ranking via binary classification scoring. A classifier predicts the probability that a page is declining; the probability becomes the score; the score makes the rank.

**Target**: `is_declining = (april_impressions < 0.8 * march_impressions)` with volume floor (March ≥ 30 impressions).

**Primary metric**: Precision@K — of the top K pages the system flags, how many are actually declining? This directly answers the business question: "of the pages we review, how many were worth reviewing?"

### Feature Vector

8 inputs, all knowable at the decision moment (2026-03-31):

| Feature | Type | NaN handling |
|---------|------|-------------|
| `log_mar_impressions` | Numeric | 0 (none missing) |
| `ctr_mar` | Numeric | 0 (when no clicks) |
| `mar_avg_position` | Numeric | fillna(0) — "no position data" |
| `momentum_feb_to_mar_pct` | Numeric | fillna(0) — 16% missing (no Feb data) |
| `engagement_rate_mar` | Numeric | fillna(0) — 51% missing (no GA4) |
| `has_clicks` | Flag | Never missing |
| `has_feb_data` | Flag | Never missing |
| `has_ga4` | Flag | Never missing |

Missingness is a signal, not noise. The `has_*` flags make the measurement gap explicit.

### Models

| Model | Why | Complexity |
|-------|-----|-----------|
| **Logistic Regression** | Readable coefficients; honest floor for "can the signals do anything?" | Low |
| **Random Forest** | Handles nonlinear interactions; compared to LR so complexity earns its place | Medium |

Both are used as probability scorers, evaluated at precision@K. The model with better precision@K at the top of the list (where editors spend time) wins for this lane.

### Baseline

Hand-crafted rule from w04:
- Risk points: 1 + 2*(CTR gap or no clicks at top-10) + 1*(spike)
- CTR shortfall: `clip(1 - ctr_ratio, 0, 1)`
- Impact: `log1p(clicks_lost) / 10`
- Score: `risk_points + ctr_shortfall + impact`
- Precision@100 = 0.750 (1.45× lift over base rate)

## Validation Design

### Client-Grouped Split

**Why grouped**: Pages from the same client share patterns (same site structure, same editorial process). A random split lets the model memorize a client and meet it again in testing, inflating the score. Week 3 measured this: random split 0.622 vs client-grouped 0.597 ROC AUC.

**How**: GroupKFold(n_splits=4) by `client_hash_id`. A fold never contains pages from clients the model trained on.

**Pooled test set**: Each page is tested exactly once. The pooled out-of-fold test set IS the full population (125,573 pages), so baseline numbers are comparable to w04.

### Metrics

| Metric | What it measures | Primary/Secondary |
|--------|-----------------|-------------------|
| Precision@K | Of top K flagged pages, how many are actually declining? | Primary (for K=10,50,100,500) |
| ROC AUC | How well does the score separate declining from non-declining? | Secondary |

## Agent Evaluation

### Planner Evaluation

- **Coverage**: Does the Planner generate hypotheses for all pages?
- **Relevance**: Are the hypotheses relevant to the page's signals?
- **Breadth**: Does the Planner generate multiple distinct hypotheses?

### Investigator Evaluation

- **Query correctness**: Do the queries return valid results?
- **Evidence classification**: Are SUPPORTS/CONTRADICTS/INCONCLUSIVE verdicts accurate?
- **Completeness**: Does the Investigator test all hypotheses?

### Synthesizer Evaluation

- **Report quality**: Is the report 3-5 sentences, specific, cited?
- **Action clarity**: Does the report recommend a specific action?
- **Honesty**: Does the report avoid causal claims?

### End-to-End Evaluation

Run on 10-20 sample pages from the w04 baseline queue. Review:
1. Does the agent pipeline complete without errors?
2. Are the reports readable and useful?
3. Do the reports cite real evidence?
4. Would an editor trust these reports?

## Leakage Controls

1. **Timeline check**: All features strictly before the label window (March → April)
2. **No label-derived features**: `trend_direction` and `trend_pct` excluded
3. **No product flags**: `health_score`, `priority_score` not in dataset
4. **Client-grouped validation**: No client appears in both train and test
5. **Baseline reproducibility**: Rule reproduced on full population before comparison
6. **Agent data access**: Investigator queries only February/March data, never April
