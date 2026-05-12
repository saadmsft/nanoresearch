---
title: "NanoResearch"
layout: default
---

<p align="center">
  <strong>A tri-level co-evolving multi-agent research automation system.</strong><br/>
  <em>Re-implementation of <a href="https://arxiv.org/abs/2605.10813">arXiv:2605.10813</a> with a ChatGPT-style web UI and field-agnostic prompts.</em>
</p>

<p align="center">
  <a href="https://github.com/saadmsft/nanoresearch">GitHub</a> •
  <a href="https://arxiv.org/abs/2605.10813">Original paper</a> •
  <a href="getting-started.html">Quickstart</a>
</p>

---

## What is NanoResearch?

NanoResearch takes a one-line research idea — in **any scholarly field** — and rides
it through ideation, planning, experimentation, analysis, writing, and review to
produce a downloadable LaTeX paper, while **learning your preferences** so the
next run feels more like _you_.

It is built around three persistent, per-user stores that co-evolve over time:

| Store | What it holds | Updated when |
|---|---|---|
| **Skill Bank** `𝒮` | Reusable procedural rules (_"design one-factor ablations"_) | Every stage end (distillation) |
| **Memory Module** `ℳ` | Project-specific facts (_"for PubMedQA, baselines are X/Y/Z"_) | Every stage end (distillation) |
| **Planner LoRA** `π_θ` | A per-user adapter on Qwen2.5-7B | Every feedback round (SDPO, Eq. 14-15) |

## Pipeline

![Pipeline overview](assets/diagrams/pipeline.png)

[View the source diagram](architecture.html#pipeline-overview)

## Documentation index

- 🚀 [Getting started](getting-started.html)
- 🏛 [Architecture deep-dive](architecture.html)
- 📜 [Paper → code mapping](paper-mapping.html)
- 🧠 [SDPO explained](sdpo.html)
- ⚡ [API reference](api.html)
- 💬 [Chat & UX design](ui-design.html)
- 🧪 [Testing guide](testing.html)
- 🔐 [Security & sandbox notes](security.html)
