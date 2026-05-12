"""Stage III — Writing & Review (paper §3.2.3).

Assembles a LaTeX paper section-by-section (to dodge context-length limits)
and runs a peer-review revision loop. Compiles to PDF via ``pdflatex`` when
available; otherwise the ``.tex`` source still ships as a final artefact.
"""

from __future__ import annotations

import re
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from pydantic import ValidationError

from ..config import get_settings
from ..llm import AgentRole, ChatMessage, Role
from ..logging import get_logger
from ..orchestrator.stage import Stage, StageContext, StageResult, StageStatus
from ..orchestrator.trajectory import Trajectory
from ._util import extract_json_object
from .artefacts import (
    AnalysisReport,
    CompiledPaper,
    PaperCritique,
    PaperDraft,
    PaperSection,
)
from .blueprint import Blueprint
from .prompts import PAPER_REVIEWER_SYSTEM, WRITING_SYSTEM

if TYPE_CHECKING:  # pragma: no cover
    from ..orchestrator.orchestrator import Orchestrator

_log = get_logger(__name__)


@dataclass
class WritingConfig:
    max_review_iterations: int = 2
    max_new_tokens_section: int = 4000
    max_new_tokens_review: int = 2500
    sections: tuple[str, ...] = (
        "introduction",
        "related_work",
        "method",
        "experiments",
        "results",
        "discussion",
        "conclusion",
    )


_LATEX_PREAMBLE = r"""\documentclass[11pt]{article}
\usepackage[margin=1in]{geometry}
\usepackage{amsmath,amssymb,amsthm}
\usepackage{graphicx}
\usepackage{booktabs}
\usepackage{hyperref}
\usepackage{xcolor}
\usepackage{microtype}
\hypersetup{colorlinks=true,linkcolor=blue,citecolor=blue,urlcolor=blue}
"""


class WritingStage(Stage):
    """Stage III — produce a LaTeX paper from Analysis + Blueprint."""

    name = "writing"
    retrieval_tags = ("writing", "paper", "latex")

    def __init__(self, *, config: WritingConfig | None = None) -> None:
        self.config = config or WritingConfig()

    def run(self, context: StageContext, orchestrator: Orchestrator) -> StageResult:
        traj = Trajectory(stage=self.name)
        blueprint: Blueprint | None = context.previous_outputs.get("blueprint")
        analysis: AnalysisReport | None = context.previous_outputs.get("analysis")
        if blueprint is None or analysis is None:
            traj.error(
                "writing_missing_inputs",
                f"need blueprint={blueprint is not None} analysis={analysis is not None}",
            )
            return StageResult(
                status=StageStatus.FAILED,
                trajectory=traj,
                summary="Cannot write without blueprint + analysis.",
            )

        # ---- 1. Draft each section ----------------------------------
        sections: list[PaperSection] = []
        for name in self.config.sections:
            body = self._write_section(
                section_name=name,
                blueprint=blueprint,
                analysis=analysis,
                prior_sections=sections,
                user_profile_summary=_profile_summary(context),
                orchestrator=orchestrator,
                traj=traj,
            )
            if body is None:
                # Don't fail the whole stage — skip and carry on.
                continue
            sections.append(PaperSection(name=name, body_latex=body))

        abstract = self._write_section(
            section_name="abstract",
            blueprint=blueprint,
            analysis=analysis,
            prior_sections=sections,
            user_profile_summary=_profile_summary(context),
            orchestrator=orchestrator,
            traj=traj,
        ) or _fallback_abstract(blueprint, analysis)

        draft = PaperDraft(
            title=blueprint.title,
            abstract_latex=abstract,
            sections=sections,
        )

        # ---- 2. Reviewer revision loop ------------------------------
        for it in range(self.config.max_review_iterations):
            critique = self._review_draft(
                draft=draft, orchestrator=orchestrator, traj=traj, iteration=it
            )
            if critique is None or critique.verdict.lower().startswith("accept"):
                traj.outcome("paper_accepted", detail=f"iterations={it}")
                break
            traj.critique(
                "paper_revise",
                detail=f"iter={it} issues={critique.issues[:3]} fixes={critique.suggested_fixes[:3]}",
            )
            draft = self._revise_draft(
                draft=draft,
                critique=critique,
                blueprint=blueprint,
                analysis=analysis,
                orchestrator=orchestrator,
                traj=traj,
                iteration=it,
            )

        # ---- 3. Compile to PDF --------------------------------------
        compiled = self._compile(draft, context.project_id, traj)

        return StageResult(
            status=StageStatus.SUCCESS,
            artefacts={
                "paper": draft,
                "compiled": compiled,
                "blueprint": blueprint,
                "analysis": analysis,
            },
            trajectory=traj,
            summary=(
                f"Wrote '{draft.title}' with {len(draft.sections)} sections; "
                + ("PDF ready" if compiled.compiled else "TeX only (no pdflatex)")
            ),
        )

    # =================================================================

    def _write_section(
        self,
        *,
        section_name: str,
        blueprint: Blueprint,
        analysis: AnalysisReport,
        prior_sections: list[PaperSection],
        user_profile_summary: str,
        orchestrator: Orchestrator,
        traj: Trajectory,
    ) -> str | None:
        # Keep prior context bounded to avoid runaway prompts.
        prior_blob = "\n\n".join(
            f"## {s.name}\n{s.body_latex[:1200]}" for s in prior_sections[-3:]
        )
        user_msg = (
            f"# Section to write: {section_name}\n\n"
            f"# Blueprint\n{blueprint.model_dump_json(indent=2)[:6000]}\n\n"
            f"# Analysis\n{analysis.model_dump_json(indent=2)[:3000]}\n\n"
            f"# User profile\n{user_profile_summary}\n\n"
            f"# Prior sections (truncated)\n{prior_blob if prior_blob else '(none)'}\n\n"
            "Return ONLY the JSON object with the body_latex for this section."
        )
        traj.action("writing_llm_call", detail=f"section={section_name}")
        res = orchestrator.router.complete(
            AgentRole.WRITING,
            [
                ChatMessage(Role.SYSTEM, WRITING_SYSTEM),
                ChatMessage(Role.USER, user_msg),
            ],
            temperature=0.4,
            max_tokens=self.config.max_new_tokens_section,
            response_format={"type": "json_object"},
        )
        raw = extract_json_object(res.text)
        if raw is None:
            traj.error("writing_no_json", detail=f"section={section_name}")
            return None
        body = raw.get("body_latex")
        if not isinstance(body, str) or not body.strip():
            traj.error("writing_empty_body", detail=f"section={section_name}")
            return None
        return body

    def _review_draft(
        self,
        *,
        draft: PaperDraft,
        orchestrator: Orchestrator,
        traj: Trajectory,
        iteration: int,
    ) -> PaperCritique | None:
        user_msg = (
            f"# Paper title\n{draft.title}\n\n"
            f"# Abstract\n{draft.abstract_latex[:1500]}\n\n"
            + "\n\n".join(
                f"## {s.name}\n{s.body_latex[:2200]}" for s in draft.sections
            )
            + "\n\nReturn ONLY the JSON object with verdict / issues / suggested_fixes."
        )
        traj.action("paper_review", detail=f"iter={iteration}")
        res = orchestrator.router.complete(
            AgentRole.REVIEW,
            [
                ChatMessage(Role.SYSTEM, PAPER_REVIEWER_SYSTEM),
                ChatMessage(Role.USER, user_msg),
            ],
            temperature=0.0,
            max_tokens=self.config.max_new_tokens_review,
            response_format={"type": "json_object"},
        )
        raw = extract_json_object(res.text)
        if raw is None:
            return None
        try:
            return PaperCritique.model_validate(raw)
        except ValidationError:
            return None

    def _revise_draft(
        self,
        *,
        draft: PaperDraft,
        critique: PaperCritique,
        blueprint: Blueprint,
        analysis: AnalysisReport,
        orchestrator: Orchestrator,
        traj: Trajectory,
        iteration: int,
    ) -> PaperDraft:
        # Heuristic: identify the sections most relevant to the reviewer's
        # complaint by keyword overlap, and rewrite those sections only.
        issues_text = " ".join(critique.issues + critique.suggested_fixes).lower()
        targeted: list[str] = []
        for s in draft.sections:
            if any(kw in issues_text for kw in (s.name, s.name.replace("_", " "))):
                targeted.append(s.name)
        if not targeted:
            targeted = [draft.sections[-1].name] if draft.sections else []

        new_sections: list[PaperSection] = []
        for s in draft.sections:
            if s.name in targeted:
                body = self._write_section(
                    section_name=f"{s.name} (revision {iteration + 1}) — address: {' | '.join(critique.issues)[:300]}",
                    blueprint=blueprint,
                    analysis=analysis,
                    prior_sections=new_sections,
                    user_profile_summary="",
                    orchestrator=orchestrator,
                    traj=traj,
                )
                new_sections.append(
                    PaperSection(name=s.name, body_latex=body or s.body_latex)
                )
            else:
                new_sections.append(s)
        return PaperDraft(
            title=draft.title,
            abstract_latex=draft.abstract_latex,
            sections=new_sections,
            bibliography_bib=draft.bibliography_bib,
        )

    # =================================================================

    def _compile(
        self,
        draft: PaperDraft,
        project_id: str,
        traj: Trajectory,
    ) -> CompiledPaper:
        papers_root = get_settings().runs_dir / "papers" / project_id
        papers_root.mkdir(parents=True, exist_ok=True)
        tex_path = papers_root / "paper.tex"
        tex_path.write_text(_assemble_tex(draft), encoding="utf-8")
        traj.action("tex_written", detail=str(tex_path))

        # Prefer pdflatex if available; fall back to tectonic (a modern,
        # single-binary alternative). Skip cleanly if neither is installed.
        pdflatex = shutil.which("pdflatex")
        tectonic = shutil.which("tectonic")
        compiler = pdflatex or tectonic
        if compiler is None:
            traj.action(
                "compile_skipped",
                detail="No LaTeX compiler on PATH (install BasicTeX / MacTeX / tectonic).",
            )
            return CompiledPaper(
                pdf_path="",
                tex_path=str(tex_path),
                compiled=False,
                compile_error="No LaTeX compiler installed",
            )

        if compiler == pdflatex:
            cmd = [pdflatex, "-interaction=nonstopmode", "-halt-on-error", "paper.tex"]
        else:
            # tectonic auto-fetches packages and compiles in one shot
            cmd = [tectonic, "paper.tex"]

        try:
            result = subprocess.run(
                cmd,
                cwd=str(papers_root),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=180,
                check=False,
            )
            ok = result.returncode == 0
            pdf_path = papers_root / "paper.pdf"
            err_tail = result.stderr.decode("utf-8", errors="replace")[-2000:]
            traj.outcome(
                "compile_done",
                detail=f"compiler={Path(compiler).name} rc={result.returncode} pdf_exists={pdf_path.exists()}",
            )
            return CompiledPaper(
                pdf_path=str(pdf_path) if ok and pdf_path.exists() else "",
                tex_path=str(tex_path),
                compiled=ok and pdf_path.exists(),
                compile_error="" if ok else err_tail,
            )
        except subprocess.TimeoutExpired:
            traj.error("compile_timeout", "LaTeX compile >180s")
            return CompiledPaper(
                pdf_path="",
                tex_path=str(tex_path),
                compiled=False,
                compile_error="LaTeX compile timeout",
            )


# =====================================================================
# helpers
# =====================================================================


def _profile_summary(context: StageContext) -> str:
    p = context.user_profile
    return (
        f"archetype={p.archetype}, domain={p.domain}, venue_style={p.venue_style or '(unspecified)'},"
        f" claim_strength={p.claim_strength or '(unspecified)'}, writing_tone={p.writing_tone or '(unspecified)'}"
    )


def _fallback_abstract(blueprint: Blueprint, analysis: AnalysisReport) -> str:
    return (
        f"This paper investigates: {blueprint.research_question}. "
        f"We propose {blueprint.proposed_method.name}: {blueprint.proposed_method.description[:600]}. "
        f"Headline finding: {analysis.headline_finding}"
    )


_SAFE_SECTION_NAME = re.compile(r"[^a-zA-Z0-9]+")


def _humanise(name: str) -> str:
    return _SAFE_SECTION_NAME.sub(" ", name).strip().title()


def _assemble_tex(draft: PaperDraft) -> str:
    parts = [
        _LATEX_PREAMBLE,
        rf"\title{{{_latex_escape_title(draft.title)}}}",
        r"\author{NanoResearch (automated)}",
        r"\date{\today}",
        r"\begin{document}",
        r"\maketitle",
        r"\begin{abstract}",
        draft.abstract_latex.strip(),
        r"\end{abstract}",
    ]
    for s in draft.sections:
        parts.append(rf"\section{{{_humanise(s.name)}}}")
        parts.append(s.body_latex.strip())
    parts.append(r"\end{document}")
    return "\n\n".join(parts) + "\n"


def _latex_escape_title(s: str) -> str:
    return (
        s.replace("\\", r"\\")
        .replace("{", r"\{")
        .replace("}", r"\}")
        .replace("$", r"\$")
        .replace("&", r"\&")
        .replace("#", r"\#")
        .replace("%", r"\%")
        .replace("_", r"\_")
    )


__all__ = ["WritingConfig", "WritingStage"]
