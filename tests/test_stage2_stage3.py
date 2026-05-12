"""Sanity tests for Stage II + III modules.

These don't hit any LLM or run real subprocesses. They verify that the
stage classes import, schemas validate, narrator covers the new events,
and the sandbox helpers can write + read files.
"""

from __future__ import annotations

from pathlib import Path

from nanoresearch.agents import (
    AnalysisStage,
    Blueprint,
    CodeFile,
    CodingStage,
    GeneratedProject,
    PaperDraft,
    PaperSection,
    WritingStage,
)
from nanoresearch.agents.artefacts import AnalysisReport, CompiledPaper, ExecutionResult
from nanoresearch.agents.sandbox import (
    list_produced,
    reset_workspace,
    run_sandboxed,
    write_files,
)
from nanoresearch.agents.stage3_writing import _assemble_tex
from nanoresearch.api.narrator import narrate_event


def test_stage_classes_importable() -> None:
    assert CodingStage.name == "coding"
    assert AnalysisStage.name == "analysis"
    assert WritingStage.name == "writing"


def test_generated_project_schema_roundtrip() -> None:
    p = GeneratedProject(
        files=[CodeFile(path="run.py", content="print('hi')")],
        entrypoint="run.py",
        notes="trivial",
    )
    raw = p.model_dump_json()
    restored = GeneratedProject.model_validate_json(raw)
    assert restored.entrypoint == "run.py"
    assert restored.files[0].content == "print('hi')"


def test_sandbox_writes_and_blocks_traversal(tmp_path: Path) -> None:
    files = [CodeFile(path="ok.py", content="print('ok')")]
    write_files(tmp_path, files)
    assert (tmp_path / "ok.py").exists()

    bad = [CodeFile(path="../escape.py", content="bad")]
    try:
        write_files(tmp_path, bad)
    except ValueError as e:
        assert "outside workspace" in str(e)
    else:
        raise AssertionError("expected ValueError for path traversal")


def test_sandbox_runs_trivial_script_and_finds_result_json(tmp_path: Path) -> None:
    reset_workspace(tmp_path)
    write_files(
        tmp_path,
        [
            CodeFile(
                path="run.py",
                content="import json\nprint('RESULT_JSON: ' + json.dumps({'accuracy': 0.9}))\n",
            )
        ],
    )
    result = run_sandboxed(workspace=tmp_path, entrypoint="run.py", timeout_seconds=10)
    assert result.success
    assert "RESULT_JSON" in result.stdout_tail
    assert result.exit_code == 0


def test_list_produced_returns_new_files_only(tmp_path: Path) -> None:
    (tmp_path / "old.txt").write_text("old")
    import time as _t
    before = _t.time()
    _t.sleep(0.05)
    (tmp_path / "new.txt").write_text("new")
    produced = list_produced(tmp_path, before)
    assert "new.txt" in produced
    assert "old.txt" not in produced


def test_narrator_covers_stage2_and_stage3_events() -> None:
    # Stage II
    s = narrate_event({"event": "trajectory_event", "kind": "action", "label": "sandbox_run", "detail": "iter=0"})
    assert s and "Running the experiment" in s

    s = narrate_event({"event": "trajectory_event", "kind": "outcome", "label": "sandbox_result", "detail": "ok=True"})
    assert s and "Run finished" in s

    # Stage III
    s = narrate_event({"event": "trajectory_event", "kind": "action", "label": "writing_llm_call", "detail": "section=method"})
    assert s and "method" in s

    # paper_ready (PDF available)
    s = narrate_event({
        "event": "paper_ready",
        "run_id": "run-abc",
        "compiled": True,
        "pdf_path": "/tmp/paper.pdf",
        "tex_path": "/tmp/paper.tex",
    })
    assert s and "/api/runs/run-abc/paper.pdf" in s

    # paper_ready (tex only)
    s = narrate_event({
        "event": "paper_ready",
        "run_id": "run-abc",
        "compiled": False,
        "pdf_path": "",
        "tex_path": "/tmp/paper.tex",
        "compile_error": "No LaTeX compiler installed",
    })
    assert s and "/api/runs/run-abc/paper.tex" in s


def test_assemble_tex_basic() -> None:
    draft = PaperDraft(
        title="A test paper with special chars: 50% & more",
        abstract_latex="The abstract body.",
        sections=[PaperSection(name="method", body_latex="The method body.")],
    )
    tex = _assemble_tex(draft)
    assert r"\documentclass" in tex
    assert r"\maketitle" in tex
    assert r"\begin{abstract}" in tex
    assert r"\section{Method}" in tex
    assert r"\end{document}" in tex
    # & and % must be escaped inside the title
    assert r"\&" in tex and r"\%" in tex


def test_compiled_paper_schema() -> None:
    cp = CompiledPaper(pdf_path="", tex_path="/tmp/p.tex", compiled=False, compile_error="no compiler")
    assert not cp.compiled
    raw = cp.model_dump_json()
    assert CompiledPaper.model_validate_json(raw).tex_path == "/tmp/p.tex"


def test_analysis_report_and_execution_result_schemas() -> None:
    er = ExecutionResult(
        success=True, exit_code=0, duration_seconds=1.2,
        stdout_tail="hello", stderr_tail="",
    )
    assert er.success
    ar = AnalysisReport(headline_finding="It worked.")
    assert ar.headline_finding == "It worked."
