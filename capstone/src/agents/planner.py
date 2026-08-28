"""Planner Agent — rule-based hypothesis generator.

Uses LangChain's Runnable interface for composability.
No LLM call — deterministic rule engine.
"""
from dataclasses import dataclass, field
from typing import Literal
from langchain_core.runnables import Runnable, RunnableConfig


@dataclass
class Hypothesis:
    code: str
    priority: Literal["critical", "high", "medium", "low"]
    reason: str
    query_target: str  # what data the Investigator should look up


@dataclass
class PlanOutput:
    content_hash_id: str
    hypotheses: list[Hypothesis]
    context: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "content_hash_id": self.content_hash_id,
            "hypotheses": [
                {"code": h.code, "priority": h.priority, "reason": h.reason}
                for h in self.hypotheses
            ],
            "context": self.context,
        }


RULES = [
    {
        "code": "new_page_low_ctr",
        "check": lambda r, v: (
            r.get("content_age_days", 999) <= 90
            and v.get("ctr_mar", 0) < (v.get("expected_ctr", 2.0) * 0.5)
        ),
        "priority": "critical",
        "reason": (
            "New page (≤90 days) with CTR < 50% of expected. "
            "Likely missing from snippets or weak title/meta."
        ),
        "query": "position_band_ctr",
    },
    {
        "code": "position_ctr_gap",
        "check": lambda r, v: (
            v.get("ctr_mar", 0) < (v.get("expected_ctr", 2.0) * 0.5)
            and v.get("mar_avg_position", 999) < 10
        ),
        "priority": "high",
        "reason": (
            "Page ranks in top 10 but CTR < 50% of position-band average. "
            "Snippet may underperform or query intent may have shifted."
        ),
        "query": "position_band_ctr",
    },
    {
        "code": "high_ctr_spike",
        "check": lambda r, v: (
            v.get("momentum", 0) is not None
            and v.get("momentum", 0) > 200
            and v.get("ctr_mar", 0) > (v.get("expected_ctr", 2.0) * 1.5)
        ),
        "priority": "high",
        "reason": (
            "Impressions jumped >200% from Feb to Mar with above-average CTR. "
            "May be a trending query that is fading, or a landing page losing freshness."
        ),
        "query": "spike_cause",
    },
    {
        "code": "zero_click_opportunity",
        "check": lambda r, v: (
            v.get("mar_clicks", 0) == 0
            and v.get("mar_avg_position", 999) < 20
        ),
        "priority": "medium",
        "reason": (
            "Page appears in top 20 positions but earns zero clicks. "
            "Strong candidate for snippet/CTA refresh."
        ),
        "query": "zero_click_detail",
    },
    {
        "code": "no_feb_data_baseline",
        "check": lambda r, v: (
            v.get("has_feb_data", 0) == 0
            and v.get("mar_clicks", 0) == 0
        ),
        "priority": "medium",
        "reason": (
            "Page has March impressions but no February data and zero clicks. "
            "Cannot assess momentum; treat as a broad refresh candidate."
        ),
        "query": "no_feb_detail",
    },
    {
        "code": "stale_content",
        "check": lambda r, v: (
            r.get("content_age_days", 0) > 365
            and v.get("momentum", 0) is not None
            and v.get("momentum", 0) < -20
        ),
        "priority": "medium",
        "reason": (
            "Page is >1 year old with impressions declining >20% MoM. "
            "Content may be stale or outdated."
        ),
        "query": "staleness_detail",
    },
    {
        "code": "impressions_declining",
        "check": lambda r, v: (
            v.get("momentum", 0) is not None
            and -50 < v.get("momentum", 0) < -20
        ),
        "priority": "low",
        "reason": (
            "Impressions declined 20-50% from Feb to Mar. "
            "Moderate decline worth monitoring but not urgent."
        ),
        "query": "decline_detail",
    },
]


class PlannerAgent(Runnable):
    """Rule-based planner: generates hypotheses from page features.

    Implements LangChain's Runnable interface so it can be composed
    into chains with Investigator and Synthesizer.
    """

    def __init__(self):
        super().__init__()
        self.rules = RULES

    def invoke(self, input: dict, config: RunnableConfig | None = None) -> PlanOutput:
        """Generate hypotheses for a single page.

        Input: {"features": dict, "context": dict} where
            features = the 8-feature vector + raw metrics
            context = content_type, search_volume, etc.
        """
        features = input.get("features", {})
        context = input.get("context", {})
        content_hash_id = features.get("content_hash_id", "unknown")

        raw_metrics = {
            "content_age_days": context.get("content_age_days", 999),
            "ctr_mar": features.get("ctr_mar", 0),
            "mar_avg_position": features.get("mar_avg_position", 999),
            "mar_clicks": features.get("mar_clicks", 0),
            "momentum": features.get("momentum_feb_to_mar_pct"),
            "has_feb_data": features.get("has_feb_data", 0),
            "has_clicks": features.get("has_clicks", 0),
        }

        hypotheses = []
        for rule in self.rules:
            try:
                if rule["check"](raw_metrics, raw_metrics):
                    hypotheses.append(Hypothesis(
                        code=rule["code"],
                        priority=rule["priority"],
                        reason=rule["reason"],
                        query_target=rule["query"],
                    ))
            except (KeyError, TypeError, ValueError):
                continue

        if not hypotheses:
            hypotheses.append(Hypothesis(
                code="no_clear_signal",
                priority="low",
                reason="No strong decline signals detected. Page appears stable.",
                query_target="general_check",
            ))

        return PlanOutput(
            content_hash_id=content_hash_id,
            hypotheses=hypotheses,
            context={**context, **raw_metrics},
        )
