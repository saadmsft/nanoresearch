---
title: SDPO explained
---

# 🧠 SDPO — Self-Distillation Policy Optimization

[← Home](index.html)

SDPO is the algorithm that turns the user's free-form feedback `ℱ` into a
**dense, token-level training signal** for the planner — no reward model, no
preference annotations. It's introduced by Buening et al. (2026) and used by
NanoResearch (Xu et al., 2026 §3.3.2) for per-user planner specialisation.

## The setup

We have:

- A planner $\pi_\theta$ — implemented here as Qwen2.5-7B with a per-user LoRA adapter.
- An input `x` — the orchestrator's prompt (system + retrieved skills/memories + topic).
- A response `y` ∼ $\pi_\theta(\cdot \mid x)$ — the planner's initial trajectory.
- Free-form user feedback `ℱ` — e.g., _"prefer simpler methods, drop the MeSH-term arm."_

We want $\pi_\theta$ to produce outputs **as if it had seen `ℱ`** even when `ℱ` isn't in the prompt.

## The trick

Treat the **feedback-conditioned** model as a **self-teacher** and the
**unconditioned** model as the student. Both are the same model — just two
different forward passes.

```mermaid
flowchart LR
  x[("Input x")] --> S[Student forward<br/>π_θ · | x, y_&lt;t]
  x --> Tplus["Teacher forward<br/>π_θ · | x, ℱ, y_&lt;t<br/>(no grad)"]
  S -->|log p y_t| Loss((Loss))
  Tplus -->|log p y_t| Loss
  Loss -->|gradient through<br/>LoRA only| W["Update LoRA"]
```

## The math (Eq. 14–15)

Per-token advantage (with stop-gradient on the teacher):

$$
A_t^{\mathrm{SDPO}} = \log \pi_\theta(\hat{y}_t \mid x, \mathcal{F}, y_{<t}) \;-\; \log \pi_\theta(\hat{y}_t \mid x, y_{<t})
$$

Policy-gradient loss (minimise):

$$
\mathcal{L}_{\mathrm{SDPO}} = - \mathbb{E}_y \left[ \sum_t \mathrm{stop\_grad}(A_t^{\mathrm{SDPO}}) \cdot \log \pi_\theta(\hat{y}_t \mid x, y_{<t}) \right]
$$

Only the **student log-probability** carries a gradient. The teacher's
log-probability is `.detach()`ed. Together with the chain rule through the
LoRA layers, this nudges the student to produce, *without ever seeing `ℱ`*,
the same distribution that the teacher (which *did* see `ℱ`) would have
produced.

## In code

```python
# src/nanoresearch/planner/sdpo.py  (excerpt)

student_input = torch.cat([student_prompt_ids, response_ids], dim=1)
teacher_input = torch.cat([teacher_prompt_ids, response_ids], dim=1)

# 1. Student: gradient-tracked.
student_logits = model(student_input).logits
student_logp_y = log_softmax(student_logits[..., R, :], dim=-1) \
    .gather(-1, response_ids[..., None]).squeeze(-1)

# 2. Teacher: stop-gradient.
with torch.no_grad():
    teacher_logits = model(teacher_input).logits
    teacher_logp_y = log_softmax(teacher_logits[..., R, :], dim=-1) \
        .gather(-1, response_ids[..., None]).squeeze(-1)

# 3. Per-token advantage (Eq. 15).
advantage = (teacher_logp_y - student_logp_y).detach().clamp(-5, 5)

# 4. Policy-gradient loss (Eq. 14).
loss = -(advantage * student_logp_y).mean()
loss.backward()    # gradients flow only through the LoRA adapter
```

## What "feedback-conditioned" actually means

We avoid the simplest construction (_appending `ℱ` as a separate user turn_)
because many chat templates reject consecutive user messages. Instead, we
**merge** the feedback into the most recent user turn:

```text
…
[USER FEEDBACK ON PRIOR ANSWER]
Prefer simpler methods. Drop the MeSH-term arm.
[END FEEDBACK]
Internalise this feedback when answering.
```

This keeps user/assistant alternation valid for both Qwen and Llama-family
chat templates and tested with the tiny-Llama fixture in
`tests/test_sdpo.py`.

## Hyperparameters

| Param | Default | Notes |
|---|---|---|
| Learning rate | `1e-4` | LoRA only, AdamW |
| Max steps per round | `50` | One round = one batch of buffered feedback records |
| Max response tokens trained | `512` | Truncate to bound memory on M1 Max |
| Advantage clip | `±5` | Stabilises early updates |
| Gradient accumulation | `1` | Increase if VRAM-tight |
| Sequence length cap | `2048` | Left-truncate longer prompts |

## Hardware budget on Apple Silicon

On an M1 Max / 32 GB, a single LoRA forward+backward at seq-len ~512 and
rank 16 sits at roughly **~10 GB unified memory** with fp16 weights and
gradient checkpointing. Each SDPO step takes ~5–8 seconds. A 50-step
round therefore finishes in 3–7 minutes — comfortable to run after every
feedback batch.

## When does it fire?

`Orchestrator.maybe_train_planner(user_id, min_examples=N)` flushes the
feedback queue and dispatches one SDPO round. The chat exposes this as
`/train` (or the assistant suggests it when buffered feedback >= 5
records).

## Why not DPO?

DPO needs pairwise preferences — *prefer A over B*. SDPO needs only the
absolute response `y` and the feedback `ℱ`. For a research assistant where
each user gives one rich textual critique per stage, SDPO is a much
cleaner fit.
