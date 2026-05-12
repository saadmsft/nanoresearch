"""Trajectory → (Skills, Memories) distillation (paper Eq. 13).

After each stage, the Orchestrator calls :func:`distill` with the recorded
trajectory ``τ`` (actions, critiques, outcomes). The distiller asks GPT-5.1
to produce a JSON object containing zero or more new Skills and Memories,
which are then validated against :mod:`nanoresearch.schemas` and persisted.
"""

from __future__ import annotations

import json
import re
import uuid
from dataclasses import dataclass
from typing import Any

from pydantic import ValidationError

from ..llm import AgentRole, ChatMessage, LLMRouter, Role
from ..logging import get_logger
from ..schemas import Memory, Skill

_log = get_logger(__name__)


_DISTILL_PROMPT = """You are the reflection component of an autonomous research system. \
After each stage of the pipeline you read the recorded trajectory and produce \
two kinds of artefacts:

- **Skills**: compact, *project-agnostic* procedural rules ("how to do X well").
  They should be reusable across topics, not tied to the current dataset.
- **Memories**: *project-specific* facts about this run (e.g. "for UCI HAR, \
  fix the train/val split before evaluating new methods").

Rules:
1. Distill **at most 3 skills** and **at most 5 memories**.
2. If the trajectory contains nothing worth saving, return empty lists. \
   Empty output is preferable to noisy output.
3. Every field is required. Use short, declarative sentences.
4. Skills must mention *when* to apply them and the concrete *procedure*.
5. Memories must specify ``topic_scope`` so they can be retrieved later.

Return JSON with exactly this shape — no prose, no markdown:

```
{
  "skills": [
    {
      "skill_type": "planning_and_execution_rule|experiment_design_rule|coding_pattern|debugging_strategy|reviewer_alignment_rule|writing_and_planning_rule|evaluation_and_writing_rule|ideation_and_writing_rule|other",
      "name": "...",
      "when_to_apply": "...",
      "procedure": "...",
      "planning_effect": "...",
      "coding_effect": "...",
      "writing_effect": "...",
      "analysis_effect": "...",
      "review_check": "...",
      "do_not": "...",
      "tags": ["tag1", "tag2"]
    }
  ],
  "memories": [
    {
      "memory_type": "project_context|decision_history|promising_direction|writing_context|failed_hypothesis|successful_outcome|user_constraint|other",
      "source_stage": "...",
      "topic_scope": "...",
      "content": "...",
      "retrieval_rationale": "...",
      "planning_implication": "...",
      "coding_implication": "...",
      "analysis_implication": "...",
      "writing_implication": "...",
      "failure_mode_to_avoid": "...",
      "tags": ["tag1"]
    }
  ]
}
```
"""


@dataclass
class DistilledArtefacts:
    skills: list[Skill]
    memories: list[Memory]


def distill(
    *,
    router: LLMRouter,
    trajectory_summary: str,
    stage: str,
    user_id: str | None = None,
    project_id: str | None = None,
    max_tokens: int = 6000,
) -> DistilledArtefacts:
    """Ask the configured Azure model to distill ``τ`` into Skills + Memories."""
    user_msg = (
        f"# Stage: {stage}\n\n"
        f"# Trajectory\n{trajectory_summary}\n\n"
        f"Return ONLY the JSON object as specified."
    )
    result = router.complete(
        AgentRole.ANALYSIS,
        [
            ChatMessage(Role.SYSTEM, _DISTILL_PROMPT),
            ChatMessage(Role.USER, user_msg),
        ],
        max_tokens=max_tokens,
        temperature=0.2,
        response_format={"type": "json_object"},
    )
    raw = _extract_json(result.text)
    if raw is None:
        _log.warning("distill_no_json", text_preview=result.text[:300])
        return DistilledArtefacts([], [])

    skills: list[Skill] = []
    memories: list[Memory] = []

    for s_raw in raw.get("skills", []) or []:
        if not isinstance(s_raw, dict):
            _log.warning("distill_skill_not_object", value_type=type(s_raw).__name__)
            continue
        try:
            s_raw.setdefault("skill_id", f"skill-{uuid.uuid4().hex[:8]}")
            skills.append(Skill.model_validate(s_raw))
        except ValidationError as e:
            _log.warning("distill_invalid_skill", error=str(e))
    for m_raw in raw.get("memories", []) or []:
        if not isinstance(m_raw, dict):
            _log.warning("distill_memory_not_object", value_type=type(m_raw).__name__)
            continue
        try:
            m_raw.setdefault("memory_id", f"mem-{uuid.uuid4().hex[:8]}")
            if user_id is not None:
                m_raw.setdefault("user_id", user_id)
            if project_id is not None:
                m_raw.setdefault("project_id", project_id)
            memories.append(Memory.model_validate(m_raw))
        except ValidationError as e:
            _log.warning("distill_invalid_memory", error=str(e))

    _log.info(
        "distill_complete",
        stage=stage,
        n_skills=len(skills),
        n_memories=len(memories),
    )
    return DistilledArtefacts(skills, memories)


_JSON_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL)


def _extract_json(text: str) -> dict[str, Any] | None:
    """Tolerant JSON extraction — handles bare JSON, fenced JSON, or leading prose."""
    if not text:
        return None
    candidates: list[str] = []
    fence_match = _JSON_FENCE_RE.search(text)
    if fence_match:
        candidates.append(fence_match.group(1).strip())
    candidates.append(text.strip())
    # As a last resort, find the outermost {...} block.
    first_brace = text.find("{")
    last_brace = text.rfind("}")
    if first_brace != -1 and last_brace > first_brace:
        candidates.append(text[first_brace : last_brace + 1])

    for c in candidates:
        try:
            data = json.loads(c)
        except json.JSONDecodeError:
            continue
        if isinstance(data, dict):
            return data
    return None
