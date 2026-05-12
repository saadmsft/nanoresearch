"""End-to-end tests for Stage I (Ideation + Planning).

Uses a scripted LLM backend and an in-memory literature client so the test
runs entirely offline. Verifies that:

- Ideation surveys literature, generates hypotheses, runs novelty
  verification, and selects ``h*``.
- Planning produces a JSON blueprint that is then revised by the peer-review
  loop until the reviewer accepts.
- Trajectories are populated with the expected event labels.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from nanoresearch.agents import (
    Blueprint,
    Hypothesis,
    IdeationStage,
    PlanningStage,
)
from nanoresearch.agents.stage1_ideation import IdeationConfig
from nanoresearch.agents.stage1_planning import PlanningConfig
from nanoresearch.literature import Paper, SearchQuery
from nanoresearch.llm import (
    ChatMessage,
    CompletionResult,
    LLMBackend,
    LLMRouter,
)
from nanoresearch.orchestrator import Orchestrator
from nanoresearch.orchestrator.stage import StageStatus
from nanoresearch.schemas import UserProfile
from nanoresearch.stores import ProfileStore


# =========================================================================
# Test doubles
# =========================================================================


class FakeLiterature:
    """Pretends to be an :class:`OpenAlexClient`."""

    def __init__(self, papers: list[Paper]) -> None:
        self._papers = papers

    def search(self, query: SearchQuery) -> list[Paper]:
        return self._papers[: query.max_results]

    def extract_evidence(self, paper: Paper) -> list:
        return []  # the ideation stage handles empty evidence gracefully


class ScriptedBackend(LLMBackend):
    """LLM backend that returns prepared replies in order."""

    name = "scripted"

    def __init__(self, replies: list[str]) -> None:
        self.replies = list(replies)
        self.calls: list[list[ChatMessage]] = []

    def complete(self, messages: list[ChatMessage], **_: Any) -> CompletionResult:
        self.calls.append(messages)
        text = self.replies.pop(0) if self.replies else "{}"
        return CompletionResult(
            text=text,
            prompt_tokens=10,
            completion_tokens=20,
            total_tokens=30,
            finish_reason="stop",
            backend=self.name,
            model="scripted",
            latency_ms=1.0,
        )


# =========================================================================
# Fixtures
# =========================================================================


@pytest.fixture
def profile_store(tmp_path: Path) -> tuple[ProfileStore, UserProfile]:
    store = ProfileStore(tmp_path)
    profile = UserProfile(
        user_id="alice",
        archetype="ai4science_journal",
        domain="NLP",
        risk_preference="moderate",
        baseline_strictness="high",
        resource_budget="1× A100, 5 days",
        venue_style="NeurIPS/ICLR conference",
        method_preference="prefer simple, ablatable methods",
    )
    store.save(profile)
    return store, profile


@pytest.fixture
def fake_papers() -> list[Paper]:
    return [
        Paper(
            paper_id="openalex:W1",
            source="openalex",
            title="BioBERT: pre-trained biomedical language representation",
            abstract="BioBERT achieves 87.5% accuracy on PubMedQA.",
            year=2023,
            citations=1000,
        ),
        Paper(
            paper_id="openalex:W2",
            source="openalex",
            title="PubMedBERT pre-training",
            abstract="PubMedBERT achieves 92.0% accuracy on PubMedQA.",
            year=2024,
            citations=500,
        ),
    ]


# =========================================================================
# Tests
# =========================================================================


def _ideation_reply() -> str:
    return json.dumps(
        {
            "research_gaps": [
                "Lightweight adapter-based fine-tuning for biomedical QA is under-explored."
            ],
            "hypotheses": [
                {
                    "hypothesis_id": "h1",
                    "statement": "A small QA-specific adapter on top of PubMedBERT improves PubMedQA accuracy.",
                    "motivation": "PubMedBERT is strong but full fine-tuning is expensive.",
                    "expected_contribution": "Match PubMedBERT accuracy with 1% of the trainable params.",
                    "related_papers": ["openalex:W1", "openalex:W2"],
                },
                {
                    "hypothesis_id": "h2",
                    "statement": "Knowledge distillation from PubMedBERT into a 6-layer student matches its PubMedQA score.",
                    "motivation": "Compute budget is tight for full PubMedBERT.",
                    "expected_contribution": "2× speedup with <2pp accuracy drop.",
                    "related_papers": ["openalex:W2"],
                },
            ],
        }
    )


def _novelty_reply() -> str:
    return json.dumps(
        {
            "judgements": [
                {
                    "hypothesis_id": "h1",
                    "novelty_score": 6.0,
                    "rationale": "Adapter approach is established but biomedical specialisation is fresh.",
                    "closest_baseline": "openalex:W1",
                },
                {
                    "hypothesis_id": "h2",
                    "novelty_score": 7.5,
                    "rationale": "Distillation of PubMedBERT specifically for biomedical QA is less common.",
                    "closest_baseline": "openalex:W2",
                },
            ]
        }
    )


def _planning_reply(blueprint_id: str = "bp-test") -> str:
    return json.dumps(
        {
            "blueprint_id": blueprint_id,
            "title": "Distilled PubMedBERT for Lightweight PubMedQA",
            "research_question": "Can a 6-layer student match PubMedBERT on PubMedQA?",
            "hypothesis": "Distillation from PubMedBERT into a 6-layer student matches its PubMedQA score.",
            "proposed_method": {
                "name": "PubMedDistill-6L",
                "description": "Layer-wise distillation of PubMedBERT into a 6-layer DistilBERT-style student with task-adaptive distillation on PubMedQA.",
                "key_components": [
                    "Teacher: PubMedBERT-base",
                    "Student: 6-layer encoder with hidden 768",
                    "Loss: KL on logits + MSE on hidden states",
                ],
                "architecture": "6-layer Transformer encoder (768 hidden, 12 heads) trained on PubMedQA dev with knowledge distillation.",
            },
            "datasets": ["PubMedQA"],
            "baselines": ["PubMedBERT-base", "BioBERT-base", "DistilBERT-PubMed (no distillation)"],
            "metrics": ["accuracy", "macro_f1", "params", "wallclock_sec"],
            "ablation_groups": [
                {
                    "name": "loss-ablation",
                    "variants": ["logit_only", "logit+hidden", "logit+hidden+attn"],
                    "purpose": "Identify which distillation signals matter.",
                }
            ],
            "compute_budget": "1× A100, 5 days",
            "expected_outcome": "Within 2pp of PubMedBERT accuracy at 50% params.",
            "risks": ["Catastrophic accuracy drop if distillation diverges."],
        }
    )


def _reviewer_reply(verdict: str, issues: list[str] | None = None) -> str:
    return json.dumps(
        {
            "verdict": verdict,
            "issues": issues or [],
            "suggested_fixes": [f"fix: {i}" for i in (issues or [])],
        }
    )


# ----- ideation -----------------------------------------------------------


def test_ideation_selects_highest_novelty_hypothesis(
    profile_store: tuple[ProfileStore, UserProfile],
    fake_papers: list[Paper],
) -> None:
    store, profile = profile_store
    backend = ScriptedBackend(
        replies=[
            _ideation_reply(),   # hypothesis generation
            _novelty_reply(),    # novelty verification
            # distill at end of stage:
            json.dumps({"skills": [], "memories": []}),
        ]
    )
    router = LLMRouter(azure=backend, planner=ScriptedBackend([]))
    orch = Orchestrator(router=router, profile_store=store)

    stage = IdeationStage(
        literature=FakeLiterature(fake_papers),
        config=IdeationConfig(max_papers=2, year_from=None),
    )
    outcome = orch.run_stage(
        stage,
        user_id=profile.user_id,
        topic="lightweight PubMedQA models",
        project_id="proj-1",
    )
    assert outcome.status is StageStatus.SUCCESS
    chosen: Hypothesis = outcome.result.artefacts["chosen_hypothesis"]
    assert chosen.hypothesis_id == "h2"  # higher novelty (7.5)
    assert chosen.novelty_score == 7.5
    papers = outcome.result.artefacts["papers"]
    assert len(papers) == 2

    labels = [e.label for e in outcome.result.trajectory.events]
    assert "literature_search" in labels
    assert "hypotheses_generated" in labels
    assert "hypothesis_selected" in labels


def test_ideation_fails_gracefully_on_garbage_llm_reply(
    profile_store: tuple[ProfileStore, UserProfile],
    fake_papers: list[Paper],
) -> None:
    store, profile = profile_store
    backend = ScriptedBackend(replies=["not json"])
    router = LLMRouter(azure=backend, planner=ScriptedBackend([]))
    orch = Orchestrator(router=router, profile_store=store)
    stage = IdeationStage(
        literature=FakeLiterature(fake_papers),
        config=IdeationConfig(max_papers=2, year_from=None),
    )
    outcome = orch.run_stage(
        stage, user_id=profile.user_id, topic="t", project_id="p"
    )
    assert outcome.status is StageStatus.FAILED


# ----- planning -----------------------------------------------------------


def _make_chosen() -> Hypothesis:
    return Hypothesis(
        hypothesis_id="h2",
        statement="Distillation of PubMedBERT matches accuracy at half the params.",
        motivation="Compute budget is tight.",
        expected_contribution="Smaller, faster biomedical QA models.",
        related_papers=["openalex:W2"],
        novelty_score=7.5,
        closest_prior_work="openalex:W2",
    )


def test_planning_accepts_blueprint_on_first_review(
    profile_store: tuple[ProfileStore, UserProfile],
) -> None:
    store, profile = profile_store
    backend = ScriptedBackend(
        replies=[
            _planning_reply(),
            _reviewer_reply("accept"),
            json.dumps({"skills": [], "memories": []}),  # distill
        ]
    )
    router = LLMRouter(azure=backend, planner=ScriptedBackend([]))
    orch = Orchestrator(router=router, profile_store=store)
    stage = PlanningStage(config=PlanningConfig(max_review_iterations=3))
    outcome = orch.run_stage(
        stage,
        user_id=profile.user_id,
        topic="t",
        project_id="p",
        previous_outputs={"chosen_hypothesis": _make_chosen()},
    )
    assert outcome.status is StageStatus.SUCCESS
    bp: Blueprint = outcome.result.artefacts["blueprint"]
    assert bp.revision_count == 0
    assert "PubMedBERT-base" in bp.baselines
    assert len(bp.ablation_groups) == 1


def test_planning_runs_review_revise_loop(
    profile_store: tuple[ProfileStore, UserProfile],
) -> None:
    store, profile = profile_store
    backend = ScriptedBackend(
        replies=[
            _planning_reply("bp-init"),
            _reviewer_reply("revise", ["Need a third baseline.", "Add an attention ablation."]),
            _planning_reply("bp-init"),  # refined version
            _reviewer_reply("accept"),
            json.dumps({"skills": [], "memories": []}),  # distill
        ]
    )
    router = LLMRouter(azure=backend, planner=ScriptedBackend([]))
    orch = Orchestrator(router=router, profile_store=store)
    stage = PlanningStage(config=PlanningConfig(max_review_iterations=3))
    outcome = orch.run_stage(
        stage,
        user_id=profile.user_id,
        topic="t",
        project_id="p",
        previous_outputs={"chosen_hypothesis": _make_chosen()},
    )
    assert outcome.status is StageStatus.SUCCESS
    bp: Blueprint = outcome.result.artefacts["blueprint"]
    assert bp.revision_count == 1
    assert "Need a third baseline." in bp.reviewer_critiques

    labels = [e.label for e in outcome.result.trajectory.events]
    assert "review_llm_call" in labels
    assert "blueprint_revise" in labels
    assert "refine_llm_call" in labels
    assert "blueprint_accepted" in labels


def test_planning_fails_without_chosen_hypothesis(
    profile_store: tuple[ProfileStore, UserProfile],
) -> None:
    store, profile = profile_store
    backend = ScriptedBackend(replies=[])
    router = LLMRouter(azure=backend, planner=ScriptedBackend([]))
    orch = Orchestrator(router=router, profile_store=store)
    stage = PlanningStage()
    outcome = orch.run_stage(
        stage, user_id=profile.user_id, topic="t", project_id="p"
    )
    assert outcome.status is StageStatus.FAILED
    assert "no chosen hypothesis" in outcome.result.summary.lower()
