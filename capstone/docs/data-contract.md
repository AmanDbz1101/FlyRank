# Data Contract

## Overview

The Agentic Content Decline Investigator uses the FlyRank internship warehouse — a pseudonymized, public-safe release of search intelligence data. This document defines the tables, features, labels, and exclusions used by the agent system.

## Warehouse Tables

| Table | Rows | Grain | Description |
|-------|------|-------|-------------|
| `fact_content_daily_performance` | 78.8M | report_date × client × content | Daily search + analytics metrics (partitioned by month) |
| `dim_content` | 519,606 | one per content item | Page metadata (type, created date, search volume) |
| `dim_clients` | 104 | one per client | Client metadata (tracking start dates) |

## Time Windows

| Window | Dates | Purpose |
|--------|-------|---------|
| **Feature window** | February + March 2026 | Compute all features the model/agent uses |
| **Decision moment** | 2026-03-31 | Everything must be knowable here |
| **Label window** | April 2026 | Compute the `is_declining` outcome (never a feature) |

## Population

- **Study population**: 125,573 content pages
- **Filter**: March impressions ≥ 30, published, not deleted
- **Clients**: 44 active in March 2026
- **Base rate**: P(is_declining) = 0.5181

## Feature Vector (8 inputs)

| Feature | Source | Knowable at Decision Moment? |
|---------|--------|------------------------------|
| `log_mar_impressions` | log1p(March GSC impressions) | Yes — March's daily rows are final |
| `ctr_mar` | March clicks / impressions × 100 | Yes — both totals final at month end |
| `mar_avg_position` | Impression-weighted March position index | Yes — March positions stop accruing |
| `momentum_feb_to_mar_pct` | (Mar - Feb) / Feb × 100 | Yes — both months closed by 31 Mar |
| `engagement_rate_mar` | GA4 engaged / sessions × 100 | Yes — on ga4_data_available IS TRUE rows |
| `has_clicks` | 1 when March clicks > 0 | Yes |
| `has_feb_data` | 1 when page has February rows | Yes |
| `has_ga4` | 1 when March GA4 sessions > 0 | Yes |

## Label Definition

```python
is_declining = (april_impressions < 0.8 * march_impressions)
```

With volume floor: March impressions ≥ 30 (low-volume wiggles excluded).

## Context Fields (not features)

| Field | Purpose |
|-------|---------|
| `content_hash_id` | Join key / pseudonym — grouping and splits only, never a feature |
| `client_hash_id` | Client grouping — grouped train/test splits only, never a feature |
| `content_type` | Read for context, not learned-from |
| `search_volume` | Read for context, not learned-from |

## Excluded Fields

| Field | Why Excluded |
|-------|-------------|
| `fact_content_query_90d` | 90-day window overlaps March+April → leaks the label |
| April / future-window columns | Contain the answer; attack A in w03 shows the jump |
| `trend_direction` / `trend_pct` | Starter label source; rebuilding is circular |
| `content_updated_date` | Snapshot-as-of-build, runs to 2026-07-06 → unknowable for 82.8% of pages |
| `is_published` / `is_deleted` | Snapshot family; used only as actionability filter, never a feature |
| Product flags (`health_score`, etc.) | Not in dataset; circular if rebuilt |
| Raw query / URL / title fields | Private; not shipped in dataset |

## Agent-Specific Data Needs

The agent system extends the feature vector with:

| Field | Source | Used By |
|-------|--------|---------|
| `content_created_date` | dim_content | Investigator (staleness check) |
| `content_type` | dim_content | Planner (hypothesis generation) |
| `ctr_expected` | Computed: mean CTR of position band | Investigator (snippet gap analysis) |
| `content_age_days` | Computed: decision_moment - created_date | Planner (staleness hypothesis) |

These fields are available at the decision moment and do not introduce leakage.

## Gotchas

- **Rate columns are ×100 percentages**: `ctr_mar = 0.53` means 0.53%, not 53%
- **`avg_position = 0` means "no data"**, not rank zero
- **Position index quirk**: values below 1 exist (101K rows) — position is an index, not literal rank
- **GA4 availability is three-valued**: TRUE / FALSE / NULL — filter with `IS TRUE`, not `fillna(0)`
- **Missingness follows content_type**: keyword articles have ~100% missing keyword data
