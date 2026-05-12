"""Schemas for Stage II (Experimentation) and Stage III (Writing) artefacts."""

from __future__ import annotations

from pydantic import BaseModel, Field


# ===========================================================================
# Stage II — Coding & Execution
# ===========================================================================


class CodeFile(BaseModel):
    """One file the Coding agent wants to write into the workspace."""

    path: str = Field(..., description="Relative path under the workspace dir.")
    content: str = Field(..., description="Raw file contents (UTF-8 text).")


class GeneratedProject(BaseModel):
    """Bundle of files produced by the Coding agent for one stage attempt."""

    files: list[CodeFile]
    entrypoint: str = Field(
        "run.py",
        description="Relative path to the script invoked to execute the experiment.",
    )
    notes: str = Field(
        "",
        description="Free-form explanation of what the project does (used by Writing).",
    )


class ExecutionResult(BaseModel):
    """One subprocess execution attempt."""

    success: bool
    exit_code: int | None
    duration_seconds: float
    stdout_tail: str
    stderr_tail: str
    timed_out: bool = False
    workspace_path: str = ""
    produced_files: list[str] = Field(default_factory=list)


class DebugPatch(BaseModel):
    """LLM-produced patch for failing code."""

    rationale: str = ""
    files: list[CodeFile] = Field(default_factory=list)


class AnalysisReport(BaseModel):
    """Stage II output consumed by Stage III.

    Field-agnostic on purpose: ``results`` is free-form structured JSON.
    """

    headline_finding: str
    quantitative_results: dict[str, str | float | int] = Field(default_factory=dict)
    qualitative_findings: list[str] = Field(default_factory=list)
    limitations: list[str] = Field(default_factory=list)
    next_steps: list[str] = Field(default_factory=list)
    raw_excerpt: str = Field(
        "",
        description="A short excerpt of the raw logs/data used by Writing for traceability.",
    )


# ===========================================================================
# Stage III — Writing & Review
# ===========================================================================


class PaperSection(BaseModel):
    name: str
    """Standard names: introduction, related_work, method, experiments,
    results, discussion, conclusion. Field-specific names also accepted."""

    body_latex: str
    """Raw LaTeX body — no document preamble, no \\section header."""


class PaperDraft(BaseModel):
    title: str
    abstract_latex: str
    sections: list[PaperSection]
    bibliography_bib: str = Field("", description="Optional BibTeX entries.")


class PaperCritique(BaseModel):
    """Reviewer's verdict on a paper draft."""

    verdict: str = Field(..., description="'accept' | 'revise'")
    issues: list[str] = Field(default_factory=list)
    suggested_fixes: list[str] = Field(default_factory=list)


class CompiledPaper(BaseModel):
    """Final artefact pointer."""

    pdf_path: str = ""
    tex_path: str = ""
    compiled: bool = False
    compile_error: str = ""
