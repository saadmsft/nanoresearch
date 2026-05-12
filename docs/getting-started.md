---
title: Getting started
---

# 🚀 Getting started

[← Home](index.html)

## Prerequisites

- **Python 3.11+** (3.12 tested)
- **Node 18+** (Vite + assistant-ui)
- An **Azure OpenAI / Foundry** GPT-5.1 deployment with AAD auth
- `az login` completed locally, with the **Cognitive Services OpenAI User** role on the resource
- _(optional)_ `pdflatex` or `tectonic` for PDF compilation
- _(optional)_ Apple-Silicon Mac with **32 GB+** unified RAM for the local Qwen2.5-7B planner

## Setup

### 1. Clone + install

```bash
git clone https://github.com/saadmsft/nanoresearch.git
cd nanoresearch
python3.12 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

### 2. Configure Azure (no API keys)

```bash
cp .env.example .env
# Edit:
#   AZURE_OPENAI_ENDPOINT=https://<your-foundry>.services.ai.azure.com/
#   AZURE_OPENAI_DEPLOYMENT=gpt-5.1
#   AZURE_OPENAI_API_VERSION=2024-12-01-preview
az login
```

Verify access:

```bash
nanoresearch health --azure --no-local
```

### 3. Backend

```bash
nanoresearch serve            # http://127.0.0.1:8000
```

### 4. Frontend

```bash
cd ui
npm install
npm run dev                   # http://localhost:5173
```

Open <http://localhost:5173>. Tell the assistant your name and field, e.g.:

> _Hi! I'm Mia, an ecologist. I prefer field studies, 6-month timeline.
> Start a run on canopy cover and breeding-bird species richness in city parks._

## Optional add-ons

### Local SDPO planner

Required for the full tri-level co-evolution (the planner learns from your
feedback via SDPO). Adds ~3 GB of wheels and downloads ~15 GB of Qwen
weights.

```bash
pip install -e ".[local]"
huggingface-cli download Qwen/Qwen2.5-7B-Instruct \
  --local-dir data/models/Qwen2.5-7B-Instruct

# Quick smoke test (loads the model)
pytest -m local_model
```

### LaTeX compiler

```bash
# Heavy option (full MacTeX):
brew install --cask mactex

# Light option (single binary, recommended):
brew install tectonic
```

NanoResearch tries `pdflatex` first then falls back to `tectonic`. If
neither is installed, you'll still get a `.tex` source ready to compile
elsewhere.

## Useful CLI commands

```bash
nanoresearch settings                  # print resolved config
nanoresearch health --azure            # AAD round-trip to GPT-5.1
nanoresearch health --azure --local    # also load Qwen on MPS
nanoresearch serve --port 8001         # alternative port
nanoresearch serve --access-log         # verbose HTTP logs
```

## Inspect a run

After a run, the full audit trail is on disk:

```bash
# All events for the latest run:
ls -t runs/ | head -1 | xargs -I {} cat runs/{}/events.jsonl

# Generated project + paper:
ls runs/workspaces/proj-*/
ls runs/papers/proj-*/

# Your accumulated stores:
ls data/users/<id>/skills/
ls data/users/<id>/memories/
```

Or via the API:

```bash
curl http://127.0.0.1:8000/api/users/<id>/skills    | jq
curl http://127.0.0.1:8000/api/users/<id>/memories  | jq
curl http://127.0.0.1:8000/api/runs/<run_id>        | jq
```

## Troubleshooting

| Symptom | Fix |
|---|---|
| `Tenant provided in token does not match resource tenant` | `az login --tenant <correct-tenant>` |
| `400 BadRequest` from OpenAlex | Network blocked or proxy interfering — set `HTTPS_PROXY` if needed |
| `pdflatex not found` warning | Install `tectonic` or MacTeX — paper still ships as `.tex` |
| Stage II fails with `ModuleNotFoundError` | Generated code asked for a package outside the allow-list — debug loop should patch on retry |
| UI shows endless 404 polling | Stale `run_id` in `localStorage`. Open DevTools → Application → Local Storage → clear `nano.runId` |

See [security.html](security.html) for sandbox details and [api.html](api.html) for the full HTTP surface.
