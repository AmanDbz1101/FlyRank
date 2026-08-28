"""Synthesizer Agent — Groq LLM report generator.

Uses LangChain's ChatGroq wrapper for the LLM call.
Falls back to plain-text summary if API is unavailable.
"""
import os
import json
from dataclasses import dataclass, field
from typing import Literal
from langchain_core.runnables import Runnable, RunnableConfig
from langchain_core.messages import HumanMessage, SystemMessage


@dataclass
class ReportOutput:
    content_hash_id: str
    report: str
    action: Literal["REFRESH", "MONITOR", "NO_ACTION"]
    confidence: Literal["high", "medium", "low"]
    evidence_used: int
    raw_llm: str = ""

    def to_dict(self) -> dict:
        return {
            "content_hash_id": self.content_hash_id,
            "report": self.report,
            "action": self.action,
            "confidence": self.confidence,
            "evidence_used": self.evidence_used,
        }


SYSTEM_PROMPT = """You are a search content analyst. Given evidence from a warehouse investigation, write a 3-5 sentence decline-risk report for a content page.

RULES:
- Cite specific evidence (numbers, percentages).
- Do NOT make causal claims ("caused by", "because of"). Use directional language ("the primary driver appears to be…").
- Recommend one action: REFRESH (edit/rewrite snippet or content), MONITOR (watch for another period), or NO_ACTION (no decline detected).
- Set confidence: high (3+ supporting evidence), medium (2), low (1 or conflicting).
- Output valid JSON: {"report": "...", "action": "...", "confidence": "..."}
"""


class SynthesizerAgent(Runnable):
    """LLM-backed report generator: turns evidence into an actionable report.

    Uses LangChain's ChatGroq for Groq API access.
    Falls back to template-based report if API is unavailable.
    """

    def __init__(self, model_name: str = "qwen/qwen3.8-27b"):
        super().__init__()
        self.model_name = model_name
        self._llm = None

    def _get_llm(self):
        if self._llm is None:
            try:
                from langchain_groq import ChatGroq
                api_key = os.environ.get("GROQ_API_KEY")
                if not api_key:
                    raise ValueError("GROQ_API_KEY not set")
                self._llm = ChatGroq(
                    model=self.model_name,
                    groq_api_key=api_key,
                    temperature=0.0,
                    max_tokens=512,
                )
            except Exception as e:
                print(f"[Synthesizer] LLM init failed: {e}")
                self._llm = False  # sentinel: don't retry
        return self._llm if self._llm is not False else None

    def invoke(self, input: dict, config: RunnableConfig | None = None) -> ReportOutput:
        """Generate a report from investigation evidence.

        Input: output of InvestigatorAgent (InvestigationOutput or dict).
        """
        if hasattr(input, "to_dict"):
            inv_dict = input.to_dict()
        else:
            inv_dict = input

        content_hash_id = inv_dict["content_hash_id"]
        evidence_list = inv_dict.get("evidence", [])
        context = inv_dict.get("context", {})

        evidence_text = self._format_evidence(evidence_list)
        llm = self._get_llm()

        if llm is None:
            return self._fallback_report(content_hash_id, evidence_list, evidence_text)

        try:
            messages = [
                SystemMessage(content=SYSTEM_PROMPT),
                HumanMessage(content=self._build_prompt(evidence_text, context)),
            ]
            response = llm.invoke(messages)
            raw = response.content.strip()
            parsed = self._parse_json(raw)

            return ReportOutput(
                content_hash_id=content_hash_id,
                report=parsed["report"],
                action=parsed["action"],
                confidence=parsed["confidence"],
                evidence_used=len(evidence_list),
                raw_llm=raw,
            )
        except Exception as e:
            print(f"[Synthesizer] LLM call failed: {e}")
            return self._fallback_report(content_hash_id, evidence_list, evidence_text)

    def _format_evidence(self, evidence_list: list[dict]) -> str:
        lines = []
        for ev in evidence_list:
            status = ev["supports"]
            lines.append(f"- [{status}] {ev['summary']}")
        return "\n".join(lines) if lines else "- No evidence collected."

    def _build_prompt(self, evidence_text: str, context: dict) -> str:
        parts = ["Investigation evidence:\n" + evidence_text]
        if context:
            parts.append(f"\nPage context: {json.dumps(context, default=str)}")
        parts.append("\nWrite the decline-risk report as JSON.")
        return "\n".join(parts)

    def _parse_json(self, raw: str) -> dict:
        """Extract JSON from LLM response, handling markdown fences."""
        text = raw
        if "```json" in text:
            text = text.split("```json", 1)[1].split("```", 1)[0]
        elif "```" in text:
            text = text.split("```", 1)[1].split("```", 1)[0]
        try:
            return json.loads(text.strip())
        except json.JSONDecodeError:
            return {"report": raw, "action": "MONITOR", "confidence": "low"}

    def _fallback_report(self, cid: str, evidence_list: list, evidence_text: str) -> ReportOutput:
        """Template-based fallback when LLM is unavailable."""
        supports = [e for e in evidence_list if e.get("supports") == "SUPPORTS"]
        contradicts = [e for e in evidence_list if e.get("supports") == "CONTRADICTS"]

        if len(supports) >= 3:
            action, confidence = "REFRESH", "high"
        elif len(supports) >= 2:
            action, confidence = "REFRESH", "medium"
        elif len(supports) >= 1:
            action, confidence = "MONITOR", "medium"
        else:
            action, confidence = "NO_ACTION", "low"

        summaries = [s["summary"] for s in supports[:3]]
        report = (
            f"Decline investigation for page {cid[:8]}…: "
            + "; ".join(summaries)
            + f" Action: {action}."
        )

        return ReportOutput(
            content_hash_id=cid,
            report=report,
            action=action,
            confidence=confidence,
            evidence_used=len(evidence_list),
        )
