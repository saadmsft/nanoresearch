---
title: Paper → code mapping
---

# 📜 Paper → code mapping

[← Home](index.html)

This page maps every concept in [arXiv:2605.10813](https://arxiv.org/abs/2605.10813)
to the file (or files) that implement it. Numbers refer to sections / equations
in the paper.

| Paper symbol | Concept | Location |
|---|---|---|
| `𝒯` | User-specified research topic | `RunSnapshot.topic` |
| `𝒰` | User profile | [`schemas.UserProfile`](https://github.com/saadmsft/nanoresearch/blob/main/src/nanoresearch/schemas/__init__.py) |
| `𝒮` | Skill Bank | [`stores.SkillBank`](https://github.com/saadmsft/nanoresearch/blob/main/src/nanoresearch/stores/skill_bank.py) |
| `ℳ` | Memory Module | [`stores.MemoryStore`](https://github.com/saadmsft/nanoresearch/blob/main/src/nanoresearch/stores/memory_store.py) |
| `𝒪` | Orchestrator | [`orchestrator.Orchestrator`](https://github.com/saadmsft/nanoresearch/blob/main/src/nanoresearch/orchestrator/orchestrator.py) |
| `π_θ` | Planner | [`planner.Planner`](https://github.com/saadmsft/nanoresearch/blob/main/src/nanoresearch/planner/planner.py) (Qwen2.5-7B + LoRA) |
| `ℱ` | Free-form user feedback | `RunManager._wait_for_feedback` |
| `ℬ` | Experiment blueprint | [`agents.Blueprint`](https://github.com/saadmsft/nanoresearch/blob/main/src/nanoresearch/agents/blueprint.py) |
| `𝒲` | Generated workspace / project | [`agents.GeneratedProject`](https://github.com/saadmsft/nanoresearch/blob/main/src/nanoresearch/agents/artefacts.py) |
| `𝒜` | Analysis report | `agents.AnalysisReport` |
| `𝒫` | Final paper PDF | `agents.CompiledPaper` |
| `h*` | Selected hypothesis | `IdeationArtefacts.chosen_hypothesis_id` |
| `c_ℬ` | Reviewer critique on blueprint | `agents.BlueprintCritique` |
| `f_R` | Reviewer critique on paper | `agents.PaperCritique` |

## Equation 1 — Stage I Ideation retrieval

> $\mathcal{S}_I, \mathcal{M}_I = \mathrm{Retrieve}(\mathcal{S}, \mathcal{M} \mid \mathcal{T}, \mathcal{U})$, $\quad P_I = \mathrm{Plan}(\mathcal{T}, \mathcal{U} \mid \mathcal{S}_I, \mathcal{M}_I)$

→ [`Orchestrator.retrieve`](https://github.com/saadmsft/nanoresearch/blob/main/src/nanoresearch/orchestrator/orchestrator.py) +
[`IdeationStage.run`](https://github.com/saadmsft/nanoresearch/blob/main/src/nanoresearch/agents/stage1_ideation.py)

## Equation 2 — Stage I Planning retrieval

Same shape, conditioned on `h*` instead of `𝒯`.

→ [`PlanningStage._initial_blueprint`](https://github.com/saadmsft/nanoresearch/blob/main/src/nanoresearch/agents/stage1_planning.py)

## Equation 3 — Peer-review correction loop

> $\mathcal{B}^{(t+1)} = \mathrm{Refine}(\mathcal{B}^{(t)}, c_\mathcal{B}^{(t)}, P_P, E)$

→ [`PlanningStage._refine_blueprint`](https://github.com/saadmsft/nanoresearch/blob/main/src/nanoresearch/agents/stage1_planning.py) — runs up to `max_review_iterations` (default 3).

## Equation 4 — Skill/Memory distillation

> $\mathcal{S}, \mathcal{M} \leftarrow \mathrm{Update}(\mathcal{S}, \mathcal{M} \mid h^*, \mathcal{B}, c_\mathcal{B})$

→ [`stores.distill`](https://github.com/saadmsft/nanoresearch/blob/main/src/nanoresearch/stores/distill.py) called from
[`Orchestrator.run_stage`](https://github.com/saadmsft/nanoresearch/blob/main/src/nanoresearch/orchestrator/orchestrator.py).

## Equation 6 — Autonomous debug loop (Stage II)

> $\mathcal{W}^{(t+1)} = \mathrm{Debug}(\mathcal{W}^{(t)} \mid \mathcal{S}_C, \mathcal{M}_C)$

→ [`CodingStage.run`](https://github.com/saadmsft/nanoresearch/blob/main/src/nanoresearch/agents/stage2_coding.py) → `_request_patch` + `_apply_patch`. Capped at `max_debug_iterations` (default 3).

## Equation 7 — Analysis report

> $\mathcal{A} = \mathrm{Analyze}(R_{\mathrm{raw}}, \mathcal{B}, \mathcal{T})$

→ [`AnalysisStage`](https://github.com/saadmsft/nanoresearch/blob/main/src/nanoresearch/agents/stage2_analysis.py). First tries to recover a `RESULT_JSON:` line printed by the generated project; falls back to LLM extraction.

## Equation 10 — Paper revision loop (Stage III)

> $\mathrm{Draft}^{(t+1)} = \mathrm{Revise}(\mathrm{Draft}^{(t)}, f_R^{(t)})$

→ [`WritingStage._revise_draft`](https://github.com/saadmsft/nanoresearch/blob/main/src/nanoresearch/agents/stage3_writing.py). Targets sections whose names appear in the reviewer's issues for re-writes (keeps untouched sections stable).

## Equations 14–15 — SDPO (planner training)

> $\nabla_\theta \mathcal{L}_{\mathrm{SDPO}} = -\mathbb{E}_y \left[ \sum_t \mathbb{E}_{\hat{y}_t} A_t^{\mathrm{SDPO}}(\hat{y}_t) \nabla_\theta \log \pi_\theta(\hat{y}_t \mid x, y_{<t}) \right]$
>
> $A_t^{\mathrm{SDPO}}(\hat{y}_t) = \log \pi_\theta(\hat{y}_t \mid x, \mathcal{F}, y_{<t}) - \log \pi_\theta(\hat{y}_t \mid x, y_{<t})$

→ [`planner.sdpo.sdpo_loss`](https://github.com/saadmsft/nanoresearch/blob/main/src/nanoresearch/planner/sdpo.py). Two forward passes (with vs. without feedback ℱ), stop-grad on teacher log-probs, advantage clipping at `±5`, LoRA-only gradient flow.

See [sdpo.html](sdpo.html) for the line-by-line derivation.

## What this implementation doesn't have (yet)

| Paper concept | Status |
|---|---|
| Compliance / Novelty / Writing judges (§ 8–10) | ⬜ |
| 20-topic benchmark harness (§ 4.2) | ⬜ |
| Simulated-scientist persona runner (§ 4.2.3) | ⬜ |
| Cross-round skill / memory growth tracking (Table 4) | ⬜ |
| Per-round efficiency / cost reporting (Table 3) | ⬜ |
| SLURM submission scripts | n/a (we run locally) |
| Figure-image generation via Gemini | n/a (we keep figures schematic) |
