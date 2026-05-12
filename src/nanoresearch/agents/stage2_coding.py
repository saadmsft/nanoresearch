"""Stage II — Coding, Execution, and Debug loop (paper §3.2.2).

This stage transforms a peer-reviewed Blueprint into a *runnable* Python
project, executes it in a subprocess sandbox, and patches it through a
bounded debug loop on failure (Eq. 6).

The output artefact is an :class:`ExecutionResult` carrying the workspace
path, produced files, and tail-logs that the AnalysisStage consumes.
"""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from pydantic import ValidationError

from ..config import get_settings
from ..llm import AgentRole, ChatMessage, Role
from ..logging import get_logger
from ..orchestrator.stage import Stage, StageContext, StageResult, StageStatus
from ..orchestrator.trajectory import Trajectory
from ._util import extract_json_object, render_memories, render_skills
from .artefacts import (
    CodeFile,
    DebugPatch,
    ExecutionResult,
    GeneratedProject,
)
from .blueprint import Blueprint
from .prompts import CODING_SYSTEM, DEBUG_SYSTEM
from .sandbox import reset_workspace, run_sandboxed, write_files

if TYPE_CHECKING:  # pragma: no cover
    from ..orchestrator.orchestrator import Orchestrator

_log = get_logger(__name__)


@dataclass
class CodingConfig:
    max_debug_iterations: int = 3
    execution_timeout_seconds: int = 240
    max_new_tokens_coding: int = 8000
    max_new_tokens_debug: int = 4000
    workspaces_root: Path | None = None


class CodingStage(Stage):
    """Stage II — generate + execute + debug a small experiment project."""

    name = "coding"
    retrieval_tags = ("coding", "experiment", "debug")

    def __init__(self, *, config: CodingConfig | None = None) -> None:
        self.config = config or CodingConfig()

    def run(self, context: StageContext, orchestrator: Orchestrator) -> StageResult:
        traj = Trajectory(stage=self.name)

        blueprint: Blueprint | None = context.previous_outputs.get("blueprint")
        if blueprint is None:
            traj.error("coding_no_blueprint", "Planning must run before Coding.")
            return StageResult(
                status=StageStatus.FAILED,
                trajectory=traj,
                summary="No blueprint available.",
            )

        workspace = self._workspace_for(context.project_id)
        reset_workspace(workspace)
        traj.action("workspace_ready", detail=f"path={workspace}")

        # ---- 1. Initial code generation -----------------------------
        project = self._generate_project(
            blueprint=blueprint,
            context=context,
            orchestrator=orchestrator,
            traj=traj,
        )
        if project is None:
            return StageResult(
                status=StageStatus.FAILED,
                trajectory=traj,
                summary="Code generation failed.",
            )

        write_files(workspace, project.files)
        traj.action(
            "code_written",
            detail=f"n_files={len(project.files)} entrypoint={project.entrypoint}",
        )

        # ---- 2. Execute + debug loop (Eq. 6) -----------------------
        attempts: list[ExecutionResult] = []
        result: ExecutionResult | None = None
        for i in range(self.config.max_debug_iterations + 1):
            traj.action(
                "sandbox_run",
                detail=f"iter={i} entrypoint={project.entrypoint}",
            )
            res = run_sandboxed(
                workspace=workspace,
                entrypoint=project.entrypoint,
                timeout_seconds=self.config.execution_timeout_seconds,
            )
            attempts.append(res)
            traj.outcome(
                "sandbox_result",
                detail=f"iter={i} ok={res.success} exit={res.exit_code} dur={res.duration_seconds:.1f}s "
                f"timeout={res.timed_out} produced={len(res.produced_files)}",
            )
            if res.success:
                result = res
                break
            if i >= self.config.max_debug_iterations:
                result = res
                break

            patch = self._request_patch(
                blueprint=blueprint,
                project=project,
                last_result=res,
                orchestrator=orchestrator,
                traj=traj,
                iteration=i,
            )
            if patch is None or not patch.files:
                traj.error("debug_no_patch", detail=f"iter={i}")
                break
            project = self._apply_patch(project, patch)
            write_files(workspace, patch.files)
            traj.action(
                "patch_applied",
                detail=f"iter={i} files={[f.path for f in patch.files]}",
            )

        assert result is not None
        if not result.success:
            return StageResult(
                status=StageStatus.FAILED,
                artefacts={
                    "execution": result,
                    "project_notes": project.notes,
                    "blueprint": blueprint,
                },
                trajectory=traj,
                summary=(
                    "Coding stage exhausted debug retries — see stderr_tail."
                    if not result.timed_out
                    else "Coding stage hit the execution timeout."
                ),
            )

        return StageResult(
            status=StageStatus.SUCCESS,
            artefacts={
                "execution": result,
                "project_notes": project.notes,
                "blueprint": blueprint,
            },
            trajectory=traj,
            summary=(
                f"Ran in {result.duration_seconds:.1f}s, produced "
                f"{len(result.produced_files)} files."
            ),
        )

    # =================================================================
    # Helpers
    # =================================================================

    def _workspace_for(self, project_id: str) -> Path:
        root = self.config.workspaces_root or (get_settings().runs_dir / "workspaces")
        return root / project_id

    def _generate_project(
        self,
        *,
        blueprint: Blueprint,
        context: StageContext,
        orchestrator: Orchestrator,
        traj: Trajectory,
    ) -> GeneratedProject | None:
        user_msg = (
            "# Experiment blueprint\n"
            f"{blueprint.model_dump_json(indent=2)}\n\n"
            "# User profile\n"
            f"- archetype: {context.user_profile.archetype}\n"
            f"- domain: {context.user_profile.domain}\n"
            f"- resource_budget: {context.user_profile.resource_budget or '(unspecified)'}\n\n"
            f"# Retrieved skills\n{render_skills(context.retrieved_skills)}\n\n"
            f"# Retrieved memories\n{render_memories(context.retrieved_memories)}\n\n"
            "Generate the project now. Remember: zero network, ≤250 lines total, "
            "print a `RESULT_JSON:` line, simulate synthetic data when needed."
        )
        traj.action("coding_llm_call", detail=f"prompt_chars={len(user_msg)}")
        res = orchestrator.router.complete(
            AgentRole.CODING,
            [
                ChatMessage(Role.SYSTEM, CODING_SYSTEM),
                ChatMessage(Role.USER, user_msg),
            ],
            temperature=0.2,
            max_tokens=self.config.max_new_tokens_coding,
            response_format={"type": "json_object"},
        )
        raw = extract_json_object(res.text)
        if raw is None:
            traj.error("coding_no_json", detail=res.text[:300])
            return None
        try:
            return GeneratedProject.model_validate(raw)
        except ValidationError as e:
            traj.error("coding_invalid_project", detail=str(e))
            return None

    def _request_patch(
        self,
        *,
        blueprint: Blueprint,
        project: GeneratedProject,
        last_result: ExecutionResult,
        orchestrator: Orchestrator,
        traj: Trajectory,
        iteration: int,
    ) -> DebugPatch | None:
        snippet = (
            f"# Last execution\n"
            f"- exit_code: {last_result.exit_code}\n"
            f"- timed_out: {last_result.timed_out}\n"
            f"- duration_s: {last_result.duration_seconds:.1f}\n\n"
            f"# stderr (tail)\n```\n{last_result.stderr_tail}\n```\n\n"
            f"# stdout (tail)\n```\n{last_result.stdout_tail}\n```\n\n"
            f"# Current files\n"
        )
        files_block = "\n".join(
            f"### {f.path}\n```python\n{f.content[:3500]}\n```" for f in project.files
        )
        user_msg = snippet + files_block
        traj.action("debug_llm_call", detail=f"iter={iteration}")
        res = orchestrator.router.complete(
            AgentRole.DEBUG,
            [
                ChatMessage(Role.SYSTEM, DEBUG_SYSTEM),
                ChatMessage(Role.USER, user_msg),
            ],
            temperature=0.1,
            max_tokens=self.config.max_new_tokens_debug,
            response_format={"type": "json_object"},
        )
        raw = extract_json_object(res.text)
        if raw is None:
            return None
        try:
            return DebugPatch.model_validate(raw)
        except ValidationError:
            return None

    @staticmethod
    def _apply_patch(project: GeneratedProject, patch: DebugPatch) -> GeneratedProject:
        by_path: dict[str, CodeFile] = {f.path: f for f in project.files}
        for f in patch.files:
            by_path[f.path] = f
        return GeneratedProject(
            files=list(by_path.values()),
            entrypoint=project.entrypoint,
            notes=project.notes,
        )


__all__ = ["CodingConfig", "CodingStage"]


# Silence ``unused import`` complaints from static analysers — uuid is kept
# available for potential downstream callers that mint workspace ids.
_ = uuid
_ = json
