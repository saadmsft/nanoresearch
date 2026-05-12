---
title: Testing
---

# 🧪 Testing

[← Home](index.html)

## Running the suite

```bash
# Default — offline only, fast (under 15 s).
pytest -m "not azure and not local_model"

# Opt-in: AAD smoke (needs `az login` + GPT-5.1 deployment).
pytest -m azure

# Opt-in: Qwen MPS smoke (needs ~15 GB model on disk + .[local] extras).
pytest -m local_model
```

## Marker key

| Marker | What it gates | Default state |
|---|---|---|
| _none_ | Pure-Python, scripted-backend tests | ✅ runs |
| `azure` | Hits Azure GPT-5.1 via AAD | ⛔ skipped |
| `local_model` | Loads Qwen2.5-7B on MPS | ⛔ skipped |
| `slow` | Anything >30 s | ⛔ skipped unless requested |

## Coverage map (61 tests)

| Module | Tests |
|---|---|
| `config/` | 3 |
| `logging/` (run manifest) | 2 |
| `llm/router.py` | 4 |
| `schemas/` + `stores/retrieval.py` | 7 |
| `stores/` (SkillBank, MemoryStore, ProfileStore) | 8 |
| `stores/distill.py` | 4 |
| `orchestrator/` | 8 |
| `literature/client.py` | 4 |
| `agents/stage1_*` (Ideation + Planning) | 5 |
| `agents/stage2_*` + `stage3_*` + `sandbox.py` + narrator | 9 |
| `api/` (HTTP routes) | 7 |
| `planner/sdpo.py` | 3 _(local-model opt-in)_ |
| Smoke | 2 _(opt-in)_ |

## Writing new tests

- **Fakes over mocks.** The `ScriptedBackend` LLM (see `tests/test_router.py`)
  is reused across the suite — it just pops pre-canned strings off a list.
- **Stages**: provide a fresh `tmp_path` for `ProfileStore` so tests don't
  collide on disk; inject scripted LLMs through `LLMRouter`.
- **API**: `fastapi.testclient.TestClient` works perfectly with our
  in-process `RunManager`. Inject backends through `create_app(router=..)`.
- **SDPO**: use the `hf-internal-testing/tiny-random-LlamaForCausalLM`
  fixture — a ~5 MB model that runs LoRA training in CPU in seconds.

## Continuous integration

The repo is wired for GitHub Actions in
[`.github/workflows/ci.yml`](https://github.com/saadmsft/nanoresearch/blob/main/.github/workflows/ci.yml) — it runs the default (offline) suite plus `npm run typecheck && npm run build` on every push.
