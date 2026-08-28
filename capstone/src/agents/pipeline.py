"""Agent Pipeline — chains Planner → Investigator → Synthesizer.

Uses LangChain's RunnableSequence for composability and visualization.
The full pipeline is a single `invoke()` call that takes a page's features
and returns a complete investigation report.
"""
import time
from dataclasses import dataclass, field
from langchain_core.runnables import Runnable, RunnableSequence, RunnableConfig

from capstone.src.agents.planner import PlannerAgent, PlanOutput
from capstone.src.agents.investigator import InvestigatorAgent, InvestigationOutput
from capstone.src.agents.synthesizer import SynthesizerAgent, ReportOutput


@dataclass
class PipelineResult:
    content_hash_id: str
    plan: dict
    investigation: dict
    report: dict
    elapsed_seconds: float

    def to_dict(self) -> dict:
        return {
            "content_hash_id": self.content_hash_id,
            "plan": self.plan,
            "investigation": self.investigation,
            "report": self.report,
            "elapsed_seconds": self.elapsed_seconds,
        }


class DeclineInvestigatorPipeline(Runnable):
    """Full agent pipeline: Planner → Investigator → Synthesizer.

    Implements LangChain's Runnable interface so it can be used
    standalone or composed into larger systems.

    Usage:
        pipeline = DeclineInvestigatorPipeline()
        result = pipeline.invoke({
            "features": {...},
            "context": {...},
        })
        print(result.report)
    """

    def __init__(self, synthesizer_model: str = "qwen/qwen3.8-27b"):
        super().__init__()
        self.planner = PlannerAgent()
        self.investigator = InvestigatorAgent()
        self.synthesizer = SynthesizerAgent(model_name=synthesizer_model)
        self.chain = self.planner | self.investigator | self.synthesizer

    def invoke(self, input: dict, config: RunnableConfig | None = None) -> PipelineResult:
        """Run the full pipeline on a single page.

        Input: {"features": dict, "context": dict}
        """
        content_hash_id = input.get("features", {}).get("content_hash_id", "unknown")
        t0 = time.time()

        plan = self.planner.invoke(input, config)
        investigation = self.investigator.invoke(plan, config)
        report = self.synthesizer.invoke(investigation, config)

        elapsed = time.time() - t0

        return PipelineResult(
            content_hash_id=content_hash_id,
            plan=plan.to_dict() if hasattr(plan, "to_dict") else plan,
            investigation=investigation.to_dict() if hasattr(investigation, "to_dict") else investigation,
            report=report.to_dict() if hasattr(report, "to_dict") else report,
            elapsed_seconds=elapsed,
        )

    def get_graph(self) -> str:
        """Return a text representation of the pipeline graph."""
        return (
            "DeclineInvestigatorPipeline\n"
            "├── PlannerAgent (rule-based)\n"
            "│   └── 7 rules → hypotheses\n"
            "├── InvestigatorAgent (DuckDB)\n"
            "│   └── query + classify evidence\n"
            "└── SynthesizerAgent (Groq LLM)\n"
            "    └── generate report + action\n"
        )


def run_pipeline_on_page(
    content_hash_id: str,
    features: dict,
    context: dict,
    synthesizer_model: str = "qwen/qwen3.8-27b",
) -> PipelineResult:
    """Convenience function to run the pipeline on a single page."""
    pipeline = DeclineInvestigatorPipeline(synthesizer_model=synthesizer_model)
    return pipeline.invoke({
        "features": {**features, "content_hash_id": content_hash_id},
        "context": context,
    })
