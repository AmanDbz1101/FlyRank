"""Investigator Agent — DuckDB query executor.

Uses LangChain's Runnable interface for composability.
Runs queries against the warehouse and classifies evidence.
"""
import dataclasses
from dataclasses import dataclass, field
from typing import Literal
import pandas as pd
import numpy as np
from langchain_core.runnables import Runnable, RunnableConfig

from capstone.src.data.warehouse import query_page_detail, query_position_band_ctr


@dataclass
class Evidence:
    hypothesis_code: str
    query: str
    rows_returned: int
    supports: Literal["SUPPORTS", "CONTRADICTS", "INCONCLUSIVE"]
    summary: str


@dataclass
class InvestigationOutput:
    content_hash_id: str
    hypotheses_tested: int
    evidence_list: list[Evidence]
    context: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "content_hash_id": self.content_hash_id,
            "hypotheses_tested": self.hypotheses_tested,
            "evidence": [
                {
                    "hypothesis": e.hypothesis_code,
                    "supports": e.supports,
                    "rows_returned": e.rows_returned,
                    "summary": e.summary,
                }
                for e in self.evidence_list
            ],
            "context": self.context,
        }


class InvestigatorAgent(Runnable):
    """Query executor: tests hypotheses against warehouse data.

    Implements LangChain's Runnable interface so it can be composed
    into chains with Planner and Synthesizer.
    """

    def __init__(self):
        super().__init__()
        self._band_ctr_cache = None

    def _get_band_ctr(self) -> pd.Series:
        if self._band_ctr_cache is None:
            self._band_ctr_cache = query_position_band_ctr()
        return self._band_ctr_cache

    def invoke(self, input: dict, config: RunnableConfig | None = None) -> InvestigationOutput:
        """Test each hypothesis by querying warehouse data.

        Input: output of PlannerAgent (PlanOutput or dict).
        """
        if hasattr(input, "to_dict"):
            plan_dict = input.to_dict()
        else:
            plan_dict = input

        content_hash_id = plan_dict["content_hash_id"]
        hypotheses = plan_dict.get("hypotheses", [])
        context = plan_dict.get("context", {})

        # Use features from Planner (same values as population) + warehouse detail for April
        detail = query_page_detail(content_hash_id)
        if detail is None:
            detail = {}

        # Merge: Planner features override warehouse detail for consistency
        merged = {**detail, **context}

        evidence_list = []
        for hyp in hypotheses:
            code = hyp["code"] if isinstance(hyp, dict) else hyp.code
            ev = self._test_hypothesis(code, merged)
            evidence_list.append(ev)

        return InvestigationOutput(
            content_hash_id=content_hash_id,
            hypotheses_tested=len(hypotheses),
            evidence_list=evidence_list,
            context=merged,
        )

    def _test_hypothesis(self, code: str, detail: dict) -> Evidence:
        """Run the query for a hypothesis and classify the evidence."""
        query_map = {
            "new_page_low_ctr": self._q_new_page_low_ctr,
            "position_ctr_gap": self._q_position_ctr_gap,
            "high_ctr_spike": self._q_spike_cause,
            "zero_click_opportunity": self._q_zero_click,
            "no_feb_data_baseline": self._q_no_feb,
            "stale_content": self._q_stale,
            "impressions_declining": self._q_declining,
            "no_clear_signal": self._q_general,
        }

        fn = query_map.get(code, self._q_general)
        return fn(detail)

    def _q_new_page_low_ctr(self, d: dict) -> Evidence:
        band_ctr = self._get_band_ctr()
        mar_pos = d.get("mar_avg_position", 999)
        pos_band = (
            "<3" if mar_pos < 3 else
            "3-10" if mar_pos < 10 else
            "10-20" if mar_pos < 20 else
            "20-50" if mar_pos < 50 else "50+"
        )
        expected = band_ctr.get(pos_band, 2.0)
        actual = d.get("ctr_mar", 0)
        ratio = actual / expected if expected > 0 else 0

        if ratio < 0.5:
            supports = "SUPPORTS"
            summary = f"CTR {actual:.1f}% is {ratio:.0%} of position-band avg ({expected:.1f}%). Needs snippet work."
        else:
            supports = "CONTRADICTS"
            summary = f"CTR {actual:.1f}% is {ratio:.0%} of position-band avg ({expected:.1f}%). Snippet is adequate."

        return Evidence(
            hypothesis_code="new_page_low_ctr",
            query="position_band_ctr",
            rows_returned=1,
            supports=supports,
            summary=summary,
        )

    def _q_position_ctr_gap(self, d: dict) -> Evidence:
        band_ctr = self._get_band_ctr()
        mar_pos = d.get("mar_avg_position", 999)
        pos_band = (
            "<3" if mar_pos < 3 else
            "3-10" if mar_pos < 10 else
            "10-20" if mar_pos < 20 else
            "20-50" if mar_pos < 50 else "50+"
        )
        expected = band_ctr.get(pos_band, 2.0)
        actual = d.get("ctr_mar", 0)
        ratio = actual / expected if expected > 0 else 0

        if mar_pos < 10 and ratio < 0.5:
            supports = "SUPPORTS"
            summary = f"Top-10 page with CTR {actual:.1f}% vs band avg {expected:.1f}%. Gap is {((1-ratio)*100):.0f}%."
        else:
            supports = "INCONCLUSIVE"
            summary = f"Position {mar_pos:.1f}, CTR {actual:.1f}%, ratio {ratio:.0%}."

        return Evidence(
            hypothesis_code="position_ctr_gap",
            query="position_band_ctr",
            rows_returned=1,
            supports=supports,
            summary=summary,
        )

    def _q_spike_cause(self, d: dict) -> Evidence:
        momentum = d.get("momentum")
        mar_imp = d.get("mar_impressions", 0)
        feb_imp = d.get("feb_impressions", 0)

        if momentum is not None and momentum > 200:
            supports = "SUPPORTS"
            summary = f"Impressions jumped from {feb_imp:,.0f} to {mar_imp:,.0f} (+{momentum:.0f}%). Trend may be fading."
        else:
            supports = "CONTRADICTS"
            summary = f"Momentum is {momentum}%. No spike detected."

        return Evidence(
            hypothesis_code="high_ctr_spike",
            query="spike_cause",
            rows_returned=1,
            supports=supports,
            summary=summary,
        )

    def _q_zero_click(self, d: dict) -> Evidence:
        clicks = d.get("mar_clicks", 0)
        pos = d.get("mar_avg_position", 999)

        if clicks == 0 and pos < 20:
            supports = "SUPPORTS"
            summary = f"Zero clicks at position {pos:.1f}. Strong snippet/CTA refresh candidate."
        else:
            supports = "CONTRADICTS"
            summary = f"Clicks={clicks}, position={pos:.1f}."

        return Evidence(
            hypothesis_code="zero_click_opportunity",
            query="zero_click_detail",
            rows_returned=1,
            supports=supports,
            summary=summary,
        )

    def _q_no_feb(self, d: dict) -> Evidence:
        feb = d.get("feb_impressions", 0)
        clicks = d.get("mar_clicks", 0)

        if feb == 0 and clicks == 0:
            supports = "SUPPORTS"
            summary = "No February data, zero clicks. Cannot assess momentum."
        else:
            supports = "INCONCLUSIVE"
            summary = f"Feb impressions={feb:,.0f}, clicks={clicks}."

        return Evidence(
            hypothesis_code="no_feb_data_baseline",
            query="no_feb_detail",
            rows_returned=1,
            supports=supports,
            summary=summary,
        )

    def _q_stale(self, d: dict) -> Evidence:
        context = d  # detail has content_age_days if available
        age = context.get("content_age_days", 0)
        momentum = context.get("momentum")

        if age > 365 and momentum is not None and momentum < -20:
            supports = "SUPPORTS"
            summary = f"Page is {age} days old with {momentum:.0f}% MoM decline. Likely stale."
        else:
            supports = "INCONCLUSIVE"
            summary = f"Age={age} days, momentum={momentum}%."

        return Evidence(
            hypothesis_code="stale_content",
            query="staleness_detail",
            rows_returned=1,
            supports=supports,
            summary=summary,
        )

    def _q_declining(self, d: dict) -> Evidence:
        momentum = d.get("momentum")

        if momentum is not None and -50 < momentum < -20:
            supports = "SUPPORTS"
            summary = f"Impressions declined {momentum:.0f}% from Feb to Mar. Moderate decline."
        else:
            supports = "INCONCLUSIVE"
            summary = f"Momentum={momentum}%."

        return Evidence(
            hypothesis_code="impressions_declining",
            query="decline_detail",
            rows_returned=1,
            supports=supports,
            summary=summary,
        )

    def _q_general(self, d: dict) -> Evidence:
        mar = d.get("mar_impressions", 0)
        apr = d.get("apr_impressions", 0)
        return Evidence(
            hypothesis_code="no_clear_signal",
            query="general_check",
            rows_returned=1,
            supports="INCONCLUSIVE",
            summary=f"March={mar:,.0f}, April={apr:,.0f}. No strong signal.",
        )
