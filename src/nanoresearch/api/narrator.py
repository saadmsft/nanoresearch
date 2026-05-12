"""Plain-English narration of pipeline events.

Maps low-level trajectory events (``hypotheses_generated``, ``review_llm_call``,
…) to user-facing messages the chat renders as streaming assistant turns.

The mapping is deterministic — zero extra LLM cost — and falls back to a
generic "working on X…" line for events it doesn't recognise.
"""

from __future__ import annotations

from typing import Any


def _shorten(text: str, n: int = 200) -> str:
    text = text or ""
    return text if len(text) <= n else text[: n - 1] + "…"


# ---------------------------------------------------------------------------


def narrate_event(payload: dict[str, Any]) -> str | None:
    """Return a user-facing narration string, or ``None`` to skip.

    ``payload`` is the same dict pushed onto the SSE queue. We branch on the
    top-level ``event`` first, then on ``label`` for ``trajectory_event``.
    """
    kind = payload.get("event")

    if kind == "run_started":
        topic = payload.get("topic", "your topic")
        return f"👋 Got it — starting a research run on _{topic}_."

    if kind == "trajectory_event":
        return _narrate_trajectory(payload)

    if kind == "stage_completed":
        stage = payload.get("stage", "stage")
        summary = _shorten(str(payload.get("summary", "")), 280)
        new_skills = payload.get("new_skills", 0)
        new_memories = payload.get("new_memories", 0)
        sm_bits = []
        if new_skills:
            sm_bits.append(f"{new_skills} new skill{'s' if new_skills != 1 else ''}")
        if new_memories:
            sm_bits.append(f"{new_memories} new memor{'ies' if new_memories != 1 else 'y'}")
        sm_note = f"  ({', '.join(sm_bits)} distilled.)" if sm_bits else ""
        return f"✅ Finished **{stage}**.{sm_note}\n\n_{summary}_"

    if kind == "awaiting_feedback":
        stage = payload.get("stage", "this step")
        return (
            f"⏸ Paused at **{stage}** — what should I emphasise or change?\n"
            f"_Tip: be specific. Anything you say here trains the planner for next time._"
        )

    if kind == "feedback_received":
        return "🙏 Got your feedback — incorporating it now."

    if kind == "feedback_enqueued":
        return None  # internal bookkeeping, don't surface

    if kind == "run_completed":
        return "🎉 Research run complete. Check the side panel for the final blueprint."

    if kind == "run_failed":
        err = payload.get("error", "unknown error")
        return f"❌ Run failed: {err}"

    if kind == "paper_ready":
        compiled = bool(payload.get("compiled"))
        pdf = payload.get("pdf_path") or ""
        tex = payload.get("tex_path") or ""
        if compiled and pdf:
            run_id = payload.get("run_id", "")
            return (
                "📄 **Paper compiled.** "
                f"[Download PDF](/api/runs/{run_id}/paper.pdf) — "
                f"or the [LaTeX source](/api/runs/{run_id}/paper.tex)."
            )
        if tex:
            run_id = payload.get("run_id", "")
            note = payload.get("compile_error") or "pdflatex not installed"
            return (
                f"📄 **Paper drafted in LaTeX** ({note}). "
                f"[Download .tex source](/api/runs/{run_id}/paper.tex)."
            )
        return "📄 Paper drafted."

    if kind == "status_changed":
        return None  # status is shown by the sidebar; don't double-narrate

    return None


# ---------------------------------------------------------------------------


def _narrate_trajectory(payload: dict[str, Any]) -> str | None:
    label = payload.get("label", "")
    detail = payload.get("detail", "")
    kind = payload.get("kind", "")

    # Errors always surface as warnings in the chat.
    if kind == "error":
        return f"⚠️ {label.replace('_', ' ')}: {_shorten(detail, 240)}"

    table = _NARRATE_BY_LABEL.get(label)
    if table is None:
        return None
    return table(detail)


def _lit_search(detail: str) -> str:
    return f"🔎 Searching scholarly databases… ({_shorten(detail, 140)})"


def _lit_done(detail: str) -> str:
    return f"📚 Done. {_shorten(detail, 80)}."


def _evidence(detail: str) -> str:
    return f"📊 Extracted quantitative evidence ({_shorten(detail, 60)})."


def _ideation_call(_: str) -> str:
    return "💡 Brainstorming candidate hypotheses…"


def _hypotheses_generated(detail: str) -> str:
    return f"💡 Drafted hypotheses ({_shorten(detail, 60)}). Checking novelty next."


def _novelty_call(_: str) -> str:
    return "🧐 Comparing each hypothesis against the prior work I just retrieved…"


def _novelty_done(detail: str) -> str:
    return f"✓ Novelty check done ({_shorten(detail, 60)})."


def _hypothesis_selected(detail: str) -> str:
    # detail looks like "id=h4 novelty=6.5 statement=Instruction-style…"
    statement = detail
    for marker in ("statement=",):
        if marker in detail:
            statement = detail.split(marker, 1)[1]
            break
    return f"🎯 Going with: **{_shorten(statement, 240)}**"


def _planning_call(_: str) -> str:
    return "📐 Drafting an experiment blueprint…"


def _review_call(detail: str) -> str:
    # On iteration 0 it's the first review; later iterations mean we're revising.
    if "iteration=0" in detail:
        return "👀 Running an internal peer review of the blueprint…"
    return f"👀 Re-reviewing the revised blueprint ({_shorten(detail, 40)})."


def _blueprint_revise(detail: str) -> str:
    return f"🔁 Reviewer asked for revisions. {_shorten(detail, 280)}"


def _blueprint_accepted(detail: str) -> str:
    return f"✅ Internal reviewer accepted the blueprint ({_shorten(detail, 40)})."


def _refine_call(_: str) -> str:
    return "✏️ Revising the blueprint to address the reviewer's feedback…"


# ---- Stage II ----------------------------------------------------------

def _workspace_ready(detail: str) -> str:
    return f"🛠 Preparing the experiment workspace ({_shorten(detail, 80)})."


def _coding_call(_: str) -> str:
    return "🧪 Writing a small experiment project to test the plan…"


def _code_written(detail: str) -> str:
    return f"📝 Project ready ({_shorten(detail, 80)}). Running it now…"


def _sandbox_run(detail: str) -> str:
    return f"▶️ Running the experiment ({_shorten(detail, 80)})…"


def _sandbox_result(detail: str) -> str:
    return f"📈 Run finished ({_shorten(detail, 120)})."


def _debug_call(detail: str) -> str:
    return f"🐛 Run failed — drafting a fix ({_shorten(detail, 60)})."


def _patch_applied(detail: str) -> str:
    return f"🔧 Patch applied ({_shorten(detail, 100)}). Re-running…"


def _analysis_call(_: str) -> str:
    return "📊 Analysing results…"


# ---- Stage III ---------------------------------------------------------

def _writing_call(detail: str) -> str:
    return f"✍️ Drafting the **{_shorten(detail.replace('section=', ''), 50)}** section…"


def _paper_review(detail: str) -> str:
    return f"👓 Reviewing the paper draft ({_shorten(detail, 40)})."


def _paper_revise(detail: str) -> str:
    return f"📝 Reviewer asked for revisions ({_shorten(detail, 220)})."


def _paper_accepted(detail: str) -> str:
    return f"✅ Reviewer accepted the paper ({_shorten(detail, 40)})."


def _tex_written(_: str) -> str:
    return "📄 LaTeX assembled. Compiling…"


def _compile_done(detail: str) -> str:
    return f"🛠 Compile finished ({_shorten(detail, 80)})."


def _compile_skipped(detail: str) -> str:
    return f"ℹ️ Skipping PDF compile ({_shorten(detail, 100)})."


def _found_result_json(detail: str) -> str:
    return f"📌 Extracted structured results: `{_shorten(detail, 160)}`"


_NARRATE_BY_LABEL = {
    # Stage I
    "literature_search": _lit_search,
    "literature_search_done": _lit_done,
    "evidence_extracted": _evidence,
    "ideation_llm_call": _ideation_call,
    "hypotheses_generated": _hypotheses_generated,
    "novelty_llm_call": _novelty_call,
    "novelty_done": _novelty_done,
    "hypothesis_selected": _hypothesis_selected,
    "planning_llm_call": _planning_call,
    "review_llm_call": _review_call,
    "blueprint_revise": _blueprint_revise,
    "blueprint_accepted": _blueprint_accepted,
    "refine_llm_call": _refine_call,
    # Stage II
    "workspace_ready": _workspace_ready,
    "coding_llm_call": _coding_call,
    "code_written": _code_written,
    "sandbox_run": _sandbox_run,
    "sandbox_result": _sandbox_result,
    "debug_llm_call": _debug_call,
    "patch_applied": _patch_applied,
    "analysis_llm_call": _analysis_call,
    "found_result_json": _found_result_json,
    # Stage III
    "writing_llm_call": _writing_call,
    "paper_review": _paper_review,
    "paper_revise": _paper_revise,
    "paper_accepted": _paper_accepted,
    "tex_written": _tex_written,
    "compile_done": _compile_done,
    "compile_skipped": _compile_skipped,
}
