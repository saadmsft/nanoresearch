---
title: Architecture
---

# 🏛 Architecture

[← Home](index.html)

NanoResearch is a **5-stage pipeline** orchestrated around two persistent stores
and one trainable planner. Each stage is a subclass of `Stage` that the
`Orchestrator` invokes in sequence, pausing for user feedback at strategic
points.

## Pipeline overview

```mermaid
flowchart LR
  T[("📝 Topic")] --> O((Orchestrator))
  O -->|retrieve top-k| SB[(Skill Bank 𝒮)]
  O -->|retrieve top-k| MM[(Memory Module ℳ)]
  O --> I[Ideation]:::s1
  I --> P[Planning]:::s1
  P --> C[Coding + Execution]:::s2
  C --> A[Analysis]:::s2
  A --> W[Writing]:::s3
  W --> R[Review]:::s3
  R --> Paper[("📄 paper.pdf")]
  W -.->|distil| SB
  W -.->|distil| MM
  I -.->|narrations| U["💬 Chat"]
  P -.-> U
  C -.-> U
  A -.-> U
  W -.-> U
  U -.->|feedback ℱ| O
  classDef s1 fill:#1e3a8a,stroke:#3b82f6,color:#fff
  classDef s2 fill:#92400e,stroke:#f59e0b,color:#fff
  classDef s3 fill:#065f46,stroke:#10b981,color:#fff
```

![Pipeline overview](assets/diagrams/pipeline.png)

## System layout

```mermaid
flowchart TB
  subgraph User["👤 User"]
    Chat["💬 Chat UI<br/>(React + Vite)"]
  end

  subgraph API["⚡ FastAPI server"]
    Intent["/api/intent<br/>NL → action"]
    RunMgr["RunManager<br/>(background thread)"]
    Narr["Narrator<br/>event → English"]
    SSE["GET /stream<br/>(SSE narrations)"]
    Files["/paper.pdf<br/>/paper.tex"]
  end

  subgraph Pipe["🔬 Orchestrator"]
    direction LR
    I[Ideation] --> P[Planning]
    P --> C[Coding]
    C --> An[Analysis]
    An --> W[Writing]
  end

  subgraph Stores["💾 Per-user stores"]
    Profile[(Profile<br/>JSON)]
    Skill[(Skill Bank<br/>JSON)]
    Mem[(Memory Module<br/>JSON)]
    LoRA[(LoRA adapter<br/>safetensors)]
  end

  subgraph Models["🤖 LLM backends"]
    Azure["Azure OpenAI<br/>GPT-5.1 · AAD auth"]
    Qwen["Qwen2.5-7B<br/>local · MPS · planner only"]
  end

  Chat <-->|HTTP| Intent
  Chat <-->|EventSource| SSE
  Chat -->|download| Files
  Intent --> RunMgr
  RunMgr --> Pipe
  Pipe --> Narr --> SSE
  Pipe <-->|retrieve / distil| Stores
  Pipe -->|9/10 of calls| Azure
  Pipe -->|planner calls only| Qwen
  Qwen <-->|SDPO LoRA training| LoRA
```

![System layout](assets/diagrams/system.png)

## Co-evolution loop

```mermaid
sequenceDiagram
  autonumber
  participant U as 👤 User
  participant C as 💬 Chat
  participant O as Orchestrator
  participant SB as Skill Bank
  participant MM as Memory Module
  participant LLM as GPT-5.1
  participant FQ as Feedback Queue
  participant SDPO as SDPO Trainer
  participant Q as Qwen + LoRA

  U->>C: "Start a run on X"
  C->>O: POST /api/runs
  loop For each stage
    O->>SB: retrieve top-k skills
    O->>MM: retrieve top-k memories
    O->>Q: plan(context) [if planner stage]
    O->>LLM: stage prompt
    LLM-->>O: stage artefact
    O-->>C: trajectory + narration (SSE)
    C-->>U: "💡 Drafted hypotheses…"
    U->>C: feedback ℱ
    C->>O: POST /feedback
    O->>FQ: enqueue (x, y, ℱ)
    O->>LLM: distil trajectory
    LLM-->>O: new skills + memories
    O->>SB: persist
    O->>MM: persist
  end
  O->>SDPO: drain queue
  SDPO->>Q: update LoRA via Eq. 14-15
  Note over Q: User's preferences are now<br/>internalised in the planner.
```

![Co-evolution sequence](assets/diagrams/sequence.png)

## Per-user filesystem layout

```text
data/users/<user_id>/
├── profile.json
├── skills/
│   ├── skill-1a2b3c4d.json
│   └── …
├── memories/
│   ├── mem-9f8e7d6c.json
│   └── …
└── lora/
    └── <user_id>/
        ├── adapter_config.json
        └── adapter_model.safetensors

runs/
├── workspaces/proj-<id>/    # ← CodingStage writes here
│   ├── run.py
│   ├── analysis.py
│   └── tables/, figures/
├── papers/proj-<id>/        # ← WritingStage writes here
│   ├── paper.tex
│   └── paper.pdf            # if pdflatex/tectonic installed
└── <run-id>/
    └── events.jsonl         # full audit trail (one line per event)
```

## What runs where

| Stage | Inputs | Output artefact | LLM role |
|---|---|---|---|
| **Ideation** | Topic, profile, skills, memories | `IdeationArtefacts` (h*) | `IDEATION` |
| **Planning** | h*, profile | `Blueprint` (peer-reviewed) | `PLANNING` + `REVIEW` |
| **Coding** | Blueprint | `GeneratedProject` + `ExecutionResult` | `CODING` + `DEBUG` |
| **Analysis** | ExecutionResult | `AnalysisReport` | `ANALYSIS` |
| **Writing** | Blueprint + Analysis | `PaperDraft` + `CompiledPaper` | `WRITING` + `REVIEW` |

## Key design decisions

- **Single-process backend, threaded runs.** Simple to reason about; one
  `RunManager` owns every run via background threads pushing onto an
  `asyncio.Queue` consumed by the SSE handler.
- **JSON-per-file stores.** Skill Bank and Memory Module are
  one-file-per-entry under `data/users/<id>/`. Transparent to inspect,
  trivial to diff in git.
- **AAD-only Azure auth.** No keys in `.env`. Uses
  `DefaultAzureCredential` → `get_bearer_token_provider` for the
  `openai.AzureOpenAI` client.
- **Field-agnostic prompts.** Every stage's system prompt explicitly invites
  the LLM to adapt to the user's field (biology, social science, math, …).
- **Network-free Stage II.** Generated experiment projects must run with the
  Python stdlib + numpy and may not call out to the internet — we simulate
  small synthetic data and let the analysis surface the methodological
  finding.
