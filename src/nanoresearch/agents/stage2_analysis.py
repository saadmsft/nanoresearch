"""Stage II — Analysis (paper Eq. 7).

Turn the raw subprocess output from CodingStage into an
:class:`AnalysisReport`. Looks for a ``RESULT_JSON:`` line first; if missing,
falls back to letting the LLM extract structure from raw text.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import TYPE_CHECKING

from pydantic import ValidationError

from ..llm import AgentRole, ChatMessage, Role
from ..logging import get_logger
from ..orchestrator.stage import Stage, StageContext, StageResult, StageStatus
from ..orchestrator.trajectory import Trajectory
from ._util import extract_json_object
from .artefacts import AnalysisReport, ExecutionResult
from .prompts import ANALYSIS_SYSTEM

if TYPE_CHECKING:  # pragma: no cover
    from ..orchestrator.orchestrator import Orchestrator

_log = get_logger(__name__)

_RESULT_LINE = re.compile(r"^RESULT_JSON:\s*(\{.*\})\s*$", re.MULTILINE)


@dataclass
class AnalysisConfig:
    max_new_tokens: int = 3000


class AnalysisStage(Stage):
    name = "analysis"
    retrieval_tags = ("analysis", "results")

    def __init__(self, *, config: AnalysisConfig | None = None) -> None:
        self.config = config or AnalysisConfig()

    def run(self, context: StageContext, orchestrator: Orchestrator) -> StageResult:
        traj = Trajectory(stage=self.name)
        execution: ExecutionResult | None = context.previous_outputs.get("execution")
        if execution is None:
            traj.error("analysis_no_execution", "CodingStage must run first.")
            return StageResult(
                status=StageStatus.FAILED, trajectory=traj, summary="No execution to analyse."
            )

        raw_result_json = _scan_for_result_json(execution.stdout_tail)
        if raw_result_json is not None:
            traj.action("found_result_json", detail=str(raw_result_json)[:200])

        report = self._analyse(
            execution=execution,
            quantitative_hint=raw_result_json,
            orchestrator=orchestrator,
            traj=traj,
        )
        if report is None:
            return StageResult(
                status=StageStatus.FAILED, trajectory=traj, summary="Analysis LLM failed."
            )

        return StageResult(
            status=StageStatus.SUCCESS,
            artefacts={
                "analysis": report,
                "execution": execution,
                "blueprint": context.previous_outputs.get("blueprint"),
            },
            trajectory=traj,
            summary=report.headline_finding[:280],
        )

    # ----------------------------------------------------------------

    def _analyse(
        self,
        *,
        execution: ExecutionResult,
        quantitative_hint: dict[str, object] | None,
        orchestrator: Orchestrator,
        traj: Trajectory,
    ) -> AnalysisReport | None:
        user_msg = (
            "# Execution summary\n"
            f"- success: {execution.success}\n"
            f"- exit_code: {execution.exit_code}\n"
            f"- duration_s: {execution.duration_seconds:.1f}\n"
            f"- produced_files: {execution.produced_files}\n\n"
            f"# Extracted RESULT_JSON (if any)\n{json.dumps(quantitative_hint, indent=2) if quantitative_hint else '(none)'}\n\n"
            f"# stdout (tail)\n```\n{execution.stdout_tail}\n```\n\n"
            f"# stderr (tail)\n```\n{execution.stderr_tail}\n```\n\n"
            "Produce the AnalysisReport JSON now."
        )
        traj.action("analysis_llm_call", detail=f"prompt_chars={len(user_msg)}")
        res = orchestrator.router.complete(
            AgentRole.ANALYSIS,
            [
                ChatMessage(Role.SYSTEM, ANALYSIS_SYSTEM),
                ChatMessage(Role.USER, user_msg),
            ],
            temperature=0.2,
            max_tokens=self.config.max_new_tokens,
            response_format={"type": "json_object"},
        )
        raw = extract_json_object(res.text)
        if raw is None:
            traj.error("analysis_no_json", detail=res.text[:300])
            return None
        try:
            return AnalysisReport.model_validate(raw)
        except ValidationError as e:
            traj.error("analysis_invalid", detail=str(e))
            return None


def _scan_for_result_json(text: str) -> dict[str, object] | None:
    m = _RESULT_LINE.search(text or "")
    if not m:
        return None
    try:
        v = json.loads(m.group(1))
        return v if isinstance(v, dict) else None
    except json.JSONDecodeError:
        return None


__all__ = ["AnalysisConfig", "AnalysisStage"]
