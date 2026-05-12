"""Stage I — Ideation phase (paper §3.2.1).

Transforms a user-specified topic ``𝒯`` and profile ``𝒰`` into a single
selected hypothesis ``h*`` via:

1. Retrieve topic- and user-aligned skills/memories (done by Orchestrator).
2. Survey the literature (OpenAlex) and extract quantitative evidence ``E``.
3. Generate ``K`` hypotheses with the literature as grounding.
4. Score each for novelty against the surveyed prior work.
5. Choose the hypothesis with the highest novelty + alignment score.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from typing import TYPE_CHECKING

from pydantic import ValidationError

from ..literature import Evidence, OpenAlexClient, Paper, SearchQuery
from ..llm import AgentRole, ChatMessage, Role
from ..logging import get_logger
from ..orchestrator.stage import (
    Stage,
    StageContext,
    StageResult,
    StageStatus,
)
from ..orchestrator.trajectory import Trajectory
from ._util import (
    extract_json_object,
    render_evidence,
    render_memories,
    render_papers,
    render_skills,
)
from .blueprint import Hypothesis, IdeationArtefacts, NoveltyJudgement
from .prompts import IDEATION_SYSTEM, NOVELTY_JUDGE_SYSTEM

if TYPE_CHECKING:  # pragma: no cover
    from ..orchestrator.orchestrator import Orchestrator

_log = get_logger(__name__)


@dataclass
class IdeationConfig:
    max_papers: int = 12
    # GPT-5.1 is a reasoning model — it spends a large share of `max_tokens` on
    # internal thinking before emitting the JSON body. Budgets below ~4k tokens
    # frequently truncate the response mid-array, so we keep them generous.
    max_new_tokens_hypotheses: int = 6000
    max_new_tokens_novelty: int = 4000
    year_from: int | None = 2020


class IdeationStage(Stage):
    """Stage I — Ideation. Produces an :class:`IdeationArtefacts` artefact."""

    name = "ideation"
    retrieval_tags = ("ideation", "literature", "hypothesis")

    def __init__(
        self,
        *,
        literature: OpenAlexClient | None = None,
        config: IdeationConfig | None = None,
    ) -> None:
        self.literature = literature or OpenAlexClient()
        self.config = config or IdeationConfig()

    # =================================================================
    def run(self, context: StageContext, orchestrator: Orchestrator) -> StageResult:
        traj = Trajectory(stage=self.name)
        topic = context.topic
        profile = context.user_profile

        # ---- 1. Literature survey -----------------------------------
        papers = self._search_literature(topic, traj)
        # ---- 2. Quantitative evidence extraction --------------------
        evidence = self._extract_evidence(papers, traj)
        # ---- 3. Hypothesis generation -------------------------------
        hypotheses = self._generate_hypotheses(
            topic=topic,
            profile=profile,
            papers=papers,
            evidence=evidence,
            context=context,
            orchestrator=orchestrator,
            traj=traj,
        )
        if not hypotheses:
            traj.error("ideation_no_hypotheses", "LLM returned no parseable hypotheses")
            return StageResult(
                status=StageStatus.FAILED,
                trajectory=traj,
                summary="Ideation failed to produce hypotheses.",
            )

        # ---- 4. Novelty verification --------------------------------
        judgements = self._verify_novelty(
            hypotheses=hypotheses,
            papers=papers,
            orchestrator=orchestrator,
            traj=traj,
        )
        # Apply novelty scores back onto hypotheses.
        by_id = {h.hypothesis_id: h for h in hypotheses}
        for j in judgements:
            h = by_id.get(j.hypothesis_id)
            if h is None:
                continue
            h.novelty_score = j.novelty_score
            h.novelty_rationale = j.rationale
            h.closest_prior_work = j.closest_baseline

        # ---- 5. Choose h* -------------------------------------------
        chosen = self._select_hypothesis(hypotheses)
        traj.outcome(
            "hypothesis_selected",
            detail=f"id={chosen.hypothesis_id} novelty={chosen.novelty_score} "
            f"statement={chosen.statement[:160]}",
        )

        artefacts = IdeationArtefacts(
            plan_PI=f"Ideation plan: survey literature on '{topic}', generate "
            f"hypotheses respecting the {profile.archetype} archetype, verify novelty.",
            retrieved_papers=[p.paper_id for p in papers],
            evidence=evidence,
            hypotheses=hypotheses,
            chosen_hypothesis_id=chosen.hypothesis_id,
        )
        return StageResult(
            status=StageStatus.SUCCESS,
            artefacts={
                "ideation": artefacts,
                "chosen_hypothesis": chosen,
                "papers": papers,
            },
            trajectory=traj,
            summary=f"Selected h* = {chosen.statement[:180]}",
        )

    # ------------------------------------------------------------- steps

    def _search_literature(self, topic: str, traj: Trajectory) -> list[Paper]:
        traj.action("literature_search", detail=f"query='{topic}'")
        query = SearchQuery(
            text=topic,
            max_results=self.config.max_papers,
            year_from=self.config.year_from,
        )
        try:
            papers = self.literature.search(query)
        except Exception as e:  # noqa: BLE001
            _log.warning("literature_search_failed", error=str(e))
            traj.error("literature_search_failed", detail=str(e))
            return []
        traj.outcome("literature_search_done", detail=f"n_papers={len(papers)}")
        return papers

    def _extract_evidence(self, papers: list[Paper], traj: Trajectory) -> list[Evidence]:
        evs: list[Evidence] = []
        for p in papers:
            evs.extend(self.literature.extract_evidence(p))
        traj.action("evidence_extracted", detail=f"n_evidence={len(evs)}")
        return evs

    def _generate_hypotheses(
        self,
        *,
        topic: str,
        profile,  # UserProfile
        papers: list[Paper],
        evidence: list[Evidence],
        context: StageContext,
        orchestrator: Orchestrator,
        traj: Trajectory,
    ) -> list[Hypothesis]:
        user_msg = (
            f"# Research topic\n{topic}\n\n"
            f"# User profile\n"
            f"- archetype: {profile.archetype}\n"
            f"- domain: {profile.domain}\n"
            f"- risk_preference: {profile.risk_preference}\n"
            f"- method_preference: {profile.method_preference or '(none)'}\n"
            f"- baseline_strictness: {profile.baseline_strictness}\n"
            f"- resource_budget: {profile.resource_budget or '(unspecified)'}\n\n"
            f"# Retrieved skills\n{render_skills(context.retrieved_skills)}\n\n"
            f"# Retrieved memories\n{render_memories(context.retrieved_memories)}\n\n"
            f"# Surveyed papers\n{render_papers(papers)}\n\n"
            f"# Quantitative evidence\n{render_evidence(evidence)}\n\n"
            f"Return ONLY the JSON object as specified."
        )
        traj.action("ideation_llm_call", detail=f"prompt_chars={len(user_msg)}")
        res = orchestrator.router.complete(
            AgentRole.IDEATION,
            [
                ChatMessage(Role.SYSTEM, IDEATION_SYSTEM),
                ChatMessage(Role.USER, user_msg),
            ],
            temperature=0.4,
            max_tokens=self.config.max_new_tokens_hypotheses,
            response_format={"type": "json_object"},
        )
        raw = extract_json_object(res.text)
        if raw is None:
            traj.error("ideation_no_json", detail=res.text[:300])
            return []
        out: list[Hypothesis] = []
        for h_raw in raw.get("hypotheses", []) or []:
            if not isinstance(h_raw, dict):
                continue
            h_raw.setdefault("hypothesis_id", f"h-{uuid.uuid4().hex[:6]}")
            try:
                out.append(Hypothesis.model_validate(h_raw))
            except ValidationError as e:
                _log.warning("ideation_invalid_hypothesis", error=str(e))
        traj.outcome("hypotheses_generated", detail=f"n={len(out)}")
        return out

    def _verify_novelty(
        self,
        *,
        hypotheses: list[Hypothesis],
        papers: list[Paper],
        orchestrator: Orchestrator,
        traj: Trajectory,
    ) -> list[NoveltyJudgement]:
        if not hypotheses:
            return []
        hypotheses_json = [
            {"hypothesis_id": h.hypothesis_id, "statement": h.statement, "motivation": h.motivation}
            for h in hypotheses
        ]
        baselines_text = render_papers(papers, max_items=8)
        user_msg = (
            f"# Prior work to judge against\n{baselines_text}\n\n"
            f"# Candidate hypotheses\n{hypotheses_json}\n\n"
            "Return ONLY the JSON object as specified."
        )
        traj.action("novelty_llm_call", detail=f"n_hypotheses={len(hypotheses)}")
        res = orchestrator.router.complete(
            AgentRole.IDEATION,
            [
                ChatMessage(Role.SYSTEM, NOVELTY_JUDGE_SYSTEM),
                ChatMessage(Role.USER, user_msg),
            ],
            temperature=0.0,
            max_tokens=self.config.max_new_tokens_novelty,
            response_format={"type": "json_object"},
        )
        raw = extract_json_object(res.text)
        if raw is None:
            traj.error("novelty_no_json", detail=res.text[:300])
            return []
        out: list[NoveltyJudgement] = []
        for j_raw in raw.get("judgements", []) or []:
            if not isinstance(j_raw, dict):
                continue
            try:
                out.append(NoveltyJudgement.model_validate(j_raw))
            except ValidationError as e:
                _log.warning("ideation_invalid_judgement", error=str(e))
        traj.outcome("novelty_done", detail=f"n_judgements={len(out)}")
        return out

    @staticmethod
    def _select_hypothesis(hypotheses: list[Hypothesis]) -> Hypothesis:
        scored = [
            (h.novelty_score if h.novelty_score is not None else 0.0, idx, h)
            for idx, h in enumerate(hypotheses)
        ]
        scored.sort(key=lambda x: x[0], reverse=True)
        return scored[0][2]
