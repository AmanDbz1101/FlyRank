"""Agentic Content Decline Investigator — Streamlit App

Three tabs:
1. Research Paper — the 9-section paper (text + charts)
2. Queue Explorer — interactive table of ranked pages
3. Agent Demo — run the agent pipeline on any page
"""
import os
import sys
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

# Ensure capstone package is importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from capstone.src.data.warehouse import load_population
from capstone.src.data.features import build_feature_vector, FEATURES8
from capstone.src.models.baseline import apply_baseline
from capstone.src.models.classifier import (
    train_and_evaluate, get_feature_importances, get_lr_coefficients,
)
from capstone.src.agents.pipeline import DeclineInvestigatorPipeline
from capstone.src.evaluation.metrics import evaluate_ranking

# ── Page config ──────────────────────────────────────────────────────
st.set_page_config(
    page_title="Content Decline Investigator",
    page_icon="🔍",
    layout="wide",
)

# ── Load data (cached) ──────────────────────────────────────────────
@st.cache_data(show_spinner="Loading warehouse data…")
def load_data():
    pop = load_population()
    pop = build_feature_vector(pop)
    pop = apply_baseline(pop)
    return pop

@st.cache_data(show_spinner="Training models…")
def train_models(fv_json: str):
    fv = pd.read_json(fv_json, orient="records")
    return train_and_evaluate(fv)

@st.cache_data(show_spinner="Computing importances…")
def compute_importances(fv_json: str):
    fv = pd.read_json(fv_json, orient="records")
    return get_feature_importances(fv)

@st.cache_data(show_spinner="Computing LR coefficients…")
def compute_lr_coefs(fv_json: str):
    fv = pd.read_json(fv_json, orient="records")
    return get_lr_coefficients(fv)

# ── Sidebar ──────────────────────────────────────────────────────────
st.sidebar.title("Content Decline Investigator")
st.sidebar.markdown("**Capstone Project** — FlyRank ML Internship 2026")
st.sidebar.divider()
st.sidebar.markdown("""
**Lane**: Refresh / Content Opportunity Scoring

**Dataset**: 125K pages, 44 clients

**Decision moment**: 2026-03-31
""")

# ── Tabs ─────────────────────────────────────────────────────────────
tab_paper, tab_queue, tab_agent = st.tabs(["📄 Research Paper", "📊 Queue Explorer", "🤖 Agent Demo"])

# ══════════════════════════════════════════════════════════════════════
# TAB 1: RESEARCH PAPER
# ══════════════════════════════════════════════════════════════════════
with tab_paper:
    st.title("Agentic Content Decline Investigator")
    st.markdown("*A multi-agent system for investigating why content pages decline in search visibility*")
    st.divider()

    # Section 1
    st.header("1. Introduction")
    st.markdown("""
Content teams invest time creating pages that drive organic traffic. When traffic declines,
the question is: **which pages are actually declining, and why?**

This project builds a **multi-agent system** (Planner → Investigator → Synthesizer) that:
- Ranks 125K+ pages by decline risk
- Generates decline investigation reports for individual pages
- Recommends actions (REFRESH / MONITOR / NO_ACTION)
    """)

    # Section 2
    st.header("2. Problem Definition")
    st.markdown("""
**Task**: Ranking via binary classification scoring

**Label**: `is_declining = (april_impressions < 0.8 × march_impressions)`

**Volume floor**: March impressions ≥ 30

**Base rate**: 0.5181 (51.8% of pages are declining)

**Primary metric**: Precision@K — of the top K flagged pages, how many are actually declining?
    """)

    # Section 3
    st.header("3. Data")
    st.markdown("""
| Table | Rows | Description |
|-------|------|-------------|
| `fact_content_daily_performance` | 78.8M | Daily search + analytics metrics |
| `dim_content` | 519,606 | Page metadata |
| `dim_clients` | 104 | Client metadata |

**Population**: 125,573 pages with March impressions ≥ 30, published, not deleted.
    """)

    # Section 4
    st.header("4. Feature Engineering")
    st.markdown("""
8 features, all knowable at the decision moment (2026-03-31):
    """)

    fv = load_data()
    fv_json = fv.to_json(orient="records")

    # Feature table
    feat_df = pd.DataFrame({
        "Feature": FEATURES8,
        "Description": [
            "log1p(March GSC impressions)",
            "March clicks / impressions × 100",
            "Impression-weighted March position index",
            "(Mar - Feb) / Feb × 100",
            "GA4 engaged / sessions × 100",
            "1 when March clicks > 0",
            "1 when page has February rows",
            "1 when March GA4 sessions > 0",
        ],
        "NaN handling": [
            "None (no missing)", "0 (no clicks)", "fillna(0)",
            "fillna(0) — 16% missing", "fillna(0) — 51% missing",
            "Never missing", "Never missing", "Never missing",
        ],
    })
    st.dataframe(feat_df, use_container_width=True, hide_index=True)

    # Section 5
    st.header("5. Baseline Rule")
    st.markdown("""
Hand-crafted rule from Week 4:
- Risk points: 1 + 2×(CTR gap or no clicks at top-10) + 1×(spike)
- CTR shortfall: `clip(1 - ctr_ratio, 0, 1)`
- Impact: `log1p(clicks_lost) / 10`
- Score: `risk_points + ctr_shortfall + impact`
    """)

    # Section 6
    st.header("6. Model Results")
    st.markdown("""
**Client-grouped 4-fold validation** (no client in both train/test):
    """)

    results = train_models(fv_json)
    base_rate = fv.is_declining.mean()

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Logistic Regression")
        lr_metrics = evaluate_ranking(results["y_true"], results["lr"], base_rate=base_rate)
    with col2:
        st.subheader("Random Forest")
        rf_metrics = evaluate_ranking(results["y_true"], results["rf"], base_rate=base_rate)

    # Precision@K chart
    k_vals = [10, 50, 100, 500]
    chart_data = pd.DataFrame({
        "K": k_vals * 2,
        "Precision@K": [lr_metrics[f"P@{k}"] for k in k_vals] + [rf_metrics[f"P@{k}"] for k in k_vals],
        "Model": ["LR"] * len(k_vals) + ["RF"] * len(k_vals),
    })
    fig = px.bar(chart_data, x="K", y="Precision@K", color="Model", barmode="group",
                 title="Precision@K: LR vs RF")
    fig.add_hline(y=base_rate, line_dash="dash", line_color="gray",
                  annotation_text=f"Base rate ({base_rate:.3f})")
    st.plotly_chart(fig, use_container_width=True)

    # Section 7
    st.header("7. Error Analysis")
    st.markdown("""
The baseline rule misses Feedly articles (32% of declines) and has a modest false positive rate (~30%)
across momentum bands. The model improves precision at the top of the queue but has low overall ROC AUC.
    """)

    # Feature importance chart
    imp = compute_importances(fv_json)
    fig_imp = px.bar(imp, x="importance", y="feature", orientation="h",
                     title="Random Forest Feature Importance")
    st.plotly_chart(fig_imp, use_container_width=True)

    # Section 8
    st.header("8. Agent System")
    st.markdown("""
Three LangChain-compatible agents chained in a pipeline:

1. **Planner** (rule-based) — generates hypotheses from page features
2. **Investigator** (DuckDB) — tests hypotheses against warehouse data
3. **Synthesizer** (Groq LLM) — generates a decline-risk report

The pipeline is a `RunnableSequence` from LangChain, making it composable and visualizable.
    """)

    # Section 9
    st.header("9. Limitations")
    st.markdown("""
- **Observational, not experimental** — cannot claim causation
- **One portfolio** — may not generalize to other industries
- **Single label window** — cannot distinguish seasonality from real decay
- **Low ROC AUC** — modest overall separation, but precision@K is strong at the top
    """)

# ══════════════════════════════════════════════════════════════════════
# TAB 2: QUEUE EXPLORER
# ══════════════════════════════════════════════════════════════════════
with tab_queue:
    st.title("📊 Queue Explorer")
    st.markdown("Explore the ranked queue of pages by decline risk.")

    col1, col2, col3 = st.columns(3)
    with col1:
        sort_by = st.selectbox("Sort by", ["baseline_score", "mar_impressions", "ctr_mar", "momentum_feb_to_mar_pct"], index=0)
    with col2:
        top_n = st.slider("Show top N", 10, 500, 50)
    with col3:
        content_filter = st.selectbox("Content type", ["All"] + sorted(fv["content_type"].dropna().unique().tolist()))

    filtered = fv.copy()
    if content_filter != "All":
        filtered = filtered[filtered["content_type"] == content_filter]

    display_cols = [
        "content_hash_id", "content_type", "mar_impressions", "ctr_mar",
        "mar_avg_position", "momentum_feb_to_mar_pct", "baseline_score", "is_declining",
    ]
    queue = filtered.nlargest(top_n, sort_by)[display_cols].reset_index(drop=True)
    queue.index += 1
    st.dataframe(queue, use_container_width=True)

    st.markdown(f"**Showing {len(queue)} of {len(filtered):,} pages** (base rate: {base_rate:.3f})")

    # Distribution chart
    fig_dist = px.histogram(filtered, x="baseline_score", nbins=50, title="Baseline Score Distribution")
    fig_dist.add_vline(x=filtered.nlargest(top_n, "baseline_score")["baseline_score"].min(),
                       line_dash="dash", line_color="red",
                       annotation_text=f"Top {top_n} threshold")
    st.plotly_chart(fig_dist, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════
# TAB 3: AGENT DEMO
# ══════════════════════════════════════════════════════════════════════
with tab_agent:
    st.title("🤖 Agent Demo")
    st.markdown("Run the decline investigation pipeline on a specific page.")

    # Select a page
    top_pages = fv.nlargest(20, "baseline_score")
    page_options = {
        f"{r.content_hash_id[:16]}… ({r.content_type}, imp={r.mar_impressions:,.0f})": r.content_hash_id
        for _, r in top_pages.iterrows()
    }

    selected_label = st.selectbox("Select a page (top 20 by risk)", list(page_options.keys()))
    selected_cid = page_options[selected_label]

    if st.button("Run Investigation", type="primary"):
        row = fv[fv.content_hash_id == selected_cid].iloc[0]

        features = {
            "content_hash_id": selected_cid,
            "ctr_mar": float(row.ctr_mar),
            "mar_avg_position": float(row.mar_avg_position) if pd.notna(row.mar_avg_position) else 999,
            "mar_clicks": float(row.mar_clicks),
            "mar_impressions": float(row.mar_impressions),
            "momentum_feb_to_mar_pct": float(row.momentum_feb_to_mar_pct) if pd.notna(row.momentum_feb_to_mar_pct) else 0,
            "has_feb_data": int(row.has_feb_data),
            "has_clicks": int(row.has_clicks),
            "has_ga4": int(row.has_ga4),
        }
        context = {
            "content_type": str(row.content_type),
            "search_volume": float(row.search_volume) if pd.notna(row.search_volume) else 0,
            "content_age_days": 100,
        }

        with st.spinner("Running pipeline…"):
            pipeline = DeclineInvestigatorPipeline()
            result = pipeline.invoke({"features": features, "context": context})

        # Display results
        st.divider()

        # Plan
        st.subheader("1️⃣ Planner — Hypotheses")
        for h in result.plan["hypotheses"]:
            priority_colors = {"critical": "🔴", "high": "🟠", "medium": "🟡", "low": "🟢"}
            icon = priority_colors.get(h["priority"], "⚪")
            st.markdown(f"{icon} **{h['code']}** ({h['priority']})")
            st.caption(h["reason"])

        # Evidence
        st.subheader("2️⃣ Investigator — Evidence")
        for e in result.investigation["evidence"]:
            status_icon = {"SUPPORTS": "✅", "CONTRADICTS": "❌", "INCONCLUSIVE": "❓"}
            icon = status_icon.get(e["supports"], "❓")
            st.markdown(f"{icon} **{e['hypothesis']}**")
            st.caption(f"{e['summary']} (rows: {e['rows_returned']})")

        # Report
        st.subheader("3️⃣ Synthesizer — Report")
        action_colors = {"REFRESH": "🟧", "MONITOR": "🟨", "NO_ACTION": "🟩"}
        action_icon = action_colors.get(result.report["action"], "⬜")
        st.markdown(f"### {action_icon} Action: {result.report['action']}")
        st.markdown(f"**Confidence**: {result.report['confidence']}")
        st.markdown(result.report["report"])
        st.caption(f"⏱️ Pipeline completed in {result.elapsed_seconds:.1f}s")
