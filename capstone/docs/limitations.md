# Limitations

## What This Work Cannot Claim

### Causal Claims

This work is **observational**, not experimental. It cannot claim:
- "Refreshing a page will cause traffic to recover"
- "Updating content improves rankings"
- "The model predicts Google's algorithm changes"

The data shows **associations** between signals and decline. A controlled experiment (randomized refresh vs. no refresh) would be needed for causal claims.

### Predictive Claims

The model **ranks** pages by decline risk — it does not **predict** whether a specific page will decline. The difference:
- **Ranking**: "Page A shows stronger decline signals than page B" (supported)
- **Prediction**: "Page A will decline by 15% next month" (not supported)

### Generalization

The model is trained on **one portfolio** (55 clients, March 2026). It may not generalize to:
- Different industries or content types
- Different time periods (seasonality, algorithm changes)
- Different geographic markets

### Agent Reports

The agent system generates **decision-support** reports, not **diagnostic** reports:
- It can say "this page shows decline signals" (supported)
- It cannot say "this page declined because of X" (not supported without causal design)

The Synthesizer's language is **directional**: "the primary driver appears to be…" — not "the cause is…"

## Data Limitations

| Limitation | Impact | Mitigation |
|-----------|--------|------------|
| **Unbalanced panel** | Only 44 of 104 clients have March data; pages absent are invisible | State the population clearly: "visible, tracked pages" |
| **GA4 sparsity** | 51% of rows have no GA4 data | `has_ga4` flag + NaN fill; never pretend absence = zero |
| **Position index quirk** | Values below 1 exist (not literal rank) | Treat as index; use band comparisons, not absolute ranks |
| **Snapshot dates** | `content_updated_date` runs to 2026-07-06 | Excluded from features; use `content_created_date` instead |
| **Single label window** | April < 80% of March — one comparison window | Cannot distinguish seasonality from real decay; state this |

## Model Limitations

| Limitation | Impact | Mitigation |
|-----------|--------|------------|
| **Low ROC AUC** | LR: 0.520, RF: 0.555 — modest overall separation | Precision@K is the primary metric; the model concentrates scores at the top |
| **LR beats RF at top** | RF's complexity doesn't earn its place at P@10-100 | Report both; LR is the better tool for small review budgets |
| **Feedly article blind spot** | Model misses 32% of feedly article declines | Content type is a context field; future work could add type-specific models |
| **Launch page false positives** | Pages with huge momentum may be flagged as "spikes" | The error analysis shows this is modest (fp ~0.30 across bands) |

## Agent Limitations

| Limitation | Impact | Mitigation |
|-----------|--------|------------|
| **Rule-based Planner** | Limited hypothesis diversity | Can be upgraded to LLM-backed later |
| **Single-window queries** | Investigator only sees March data | Cannot detect multi-month trends |
| **LLM hallucination risk** | Synthesizer may generate plausible but unsupported claims | Evidence list constrains the output; review recommended |
| **Token budget** | Groq free tier has rate limits | Cache reports; limit to top-K pages |
| **No real-time data** | Warehouse is a snapshot, not live | Reports reflect March 2026 state only |

## What Would Make This Work Stronger

1. **Multi-month analysis**: Use 3+ months of data to detect persistent vs. transient decline
2. **Experiment design**: A/B test refresh recommendations to measure causal impact
3. **Client-specific models**: Train per-client models for clients with enough data
4. **Query-level analysis**: Use `fact_content_query_90d` with careful window alignment
5. **Human feedback loop**: Editor corrections retrain the Planner's hypothesis selection
