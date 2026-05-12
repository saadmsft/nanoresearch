---
title: API reference
---

# ⚡ API reference

[← Home](index.html)

NanoResearch exposes a FastAPI HTTP surface used by the UI. All endpoints are
unauthenticated and bound to `127.0.0.1` by default — put behind a reverse
proxy / auth layer before exposing.

Base URL: `http://127.0.0.1:8000/api`

## Users

### `POST /users` — upsert a profile

```json
{
  "user_id": "alice",
  "archetype": "ai4science_journal",
  "domain": "Biology",
  "risk_preference": "moderate",
  "baseline_strictness": "high",
  "persona_brief": "Prefers conservative methods…"
}
```

→ **201** `UserProfile` (full schema in [`schemas.UserProfile`](https://github.com/saadmsft/nanoresearch/blob/main/src/nanoresearch/schemas/__init__.py))

### `GET /users` → `list[str]`

Returns user_ids.

### `GET /users/{user_id}` → `UserProfile`

### `GET /users/{user_id}/skills` → `list[Skill]`

### `GET /users/{user_id}/memories` → `list[Memory]`

## Intent (chat command router)

### `POST /intent`

```json
{
  "text": "I'm Mia, ecology. Start a run on bird diversity in city parks.",
  "session": {
    "user_id": "demo",
    "run_id": null,
    "run_status": null,
    "has_profile": true
  }
}
```

→ **200**

```json
{
  "source": "local | llm",
  "intent": {
    "action": "start_run",
    "topic": "bird diversity in city parks",
    "reply": "Got it — starting on bird diversity…"
  }
}
```

`action` ∈ `help`, `create_user`, `select_user`, `update_profile`,
`list_users`, `start_run`, `submit_feedback`, `status`, `list_skills`,
`list_memories`, `train_planner`, `chitchat`.

## Runs

### `POST /runs` — start a run

```json
{ "user_id": "demo", "topic": "..." }
```

→ **201** `RunSnapshot`

```json
{
  "run_id": "run-1778624824-17764284",
  "user_id": "demo",
  "topic": "...",
  "project_id": "proj-1778624824-17764284",
  "status": "pending | running | awaiting_feedback | completed | failed",
  "current_stage": "ideation | planning | coding | analysis | writing | null",
  "stages_completed": ["ideation"],
  "last_summary": "Selected h* = …",
  "started_at": "2026-05-12T…Z",
  "updated_at": "2026-05-12T…Z",
  "error": null
}
```

### `GET /runs` → `list[RunSnapshot]`

### `GET /runs/{run_id}` → `RunSnapshot`

### `POST /runs/{run_id}/feedback`

```json
{ "text": "Drop the green wall arm — too speculative." }
```

→ **200** `RunSnapshot`  
→ **409** if the run isn't `awaiting_feedback`.

### `GET /runs/{run_id}/stream` — Server-Sent Events

Sends a stream of `RunEvent` JSON objects with these `event` types:

| Event | Payload |
|---|---|
| `run_started` | `topic` |
| `status_changed` | `status` |
| `trajectory_event` | `kind`, `label`, `detail`, `metadata` |
| `stage_completed` | `stage`, `status`, `summary`, `new_skills`, `new_memories` |
| `awaiting_feedback` | `stage` |
| `feedback_received` | `text` |
| `feedback_enqueued` | `stage` |
| `paper_ready` | `compiled`, `pdf_path`, `tex_path`, `compile_error` |
| `run_completed` | — |
| `run_failed` | `error` |
| `narration` | `text` (plain-English version for chat display) |
| `stream_end` | — (tombstone) |

### `GET /runs/{run_id}/paper.pdf`

Streams the compiled PDF. **404** if no PDF (compiler missing or
compile failed).

### `GET /runs/{run_id}/paper.tex`

Streams the LaTeX source. Always available after Stage III completes.

## Health

### `GET /health` → `{"status": "ok", "version": "0.1.0"}`

## Errors

All endpoints return either:

```json
{ "detail": "human-readable error" }
```

or, for validation failures, the standard FastAPI shape with the offending
field paths. CORS is enabled for `http://localhost:5173` only.
