---
title: Security & sandbox
---

# 🔐 Security & sandbox

[← Home](index.html)

NanoResearch executes LLM-generated Python code as part of Stage II. Read
this before exposing the API to anything other than `127.0.0.1`.

## Threat model

We assume the **LLM is honest but unreliable**:

- ✅ Won't deliberately exfiltrate data
- ⚠️ Might produce code that opens sockets, writes outside the workspace,
  spawns runaway loops, or imports forbidden packages
- ⚠️ Might be coerced via prompt injection from retrieved literature
  abstracts

We **do not** defend against a malicious LLM with internet egress; that
would require a stronger sandbox (Docker / gVisor / firejail) which is
on the [roadmap](https://github.com/saadmsft/nanoresearch#-roadmap).

## Current sandbox guarantees

[`src/nanoresearch/agents/sandbox.py`](https://github.com/saadmsft/nanoresearch/blob/main/src/nanoresearch/agents/sandbox.py) enforces:

| Layer | Mechanism | Effect |
|---|---|---|
| **Path traversal** | `write_files()` resolves each target and refuses anything outside the workspace | LLM cannot drop a file in `~/.ssh` |
| **Memory** | `RLIMIT_AS = 2 GB` | OOM-killed after 2 GB |
| **CPU time** | `RLIMIT_CPU` and `subprocess.run(timeout=…)` | Killed after 240 s by default |
| **Network** | `HTTP_PROXY`, `HTTPS_PROXY` env stripped; `NO_PROXY=*` | DNS still resolves; doesn't block raw sockets |
| **Filesystem** | `cwd` set to workspace directory | Relative paths land inside the sandbox |
| **Environment** | `PYTHONSTARTUP`, proxies removed; clean child env | No shell escape via dotfiles |
| **Imports** | Coding prompt restricts allowed packages | Soft guarantee — debug loop catches violations |

## What's NOT blocked

- Raw socket calls (`socket.socket(...).connect(...)`) on a network that
  isn't proxied
- Filesystem reads outside the workspace (e.g., `open('/etc/passwd')`) —
  Stage II output is captured and analysed, but the read itself happens
- Forks / subprocesses spawned by the generated code (they inherit the
  RLIMITs but aren't traced)

**Do not run NanoResearch on a machine with secrets that the local user
shouldn't see.** Run on a personal dev box or inside a VM.

## Prompt injection from literature

OpenAlex abstracts are inlined into the Ideation prompt. A maliciously
crafted abstract can attempt to redirect the model. We mitigate by:

- Truncating each abstract to ~280 chars in the rendered prompt block
- Always instructing the LLM to "DO NOT copy retrieved content — they
  are general patterns"
- Returning **only** JSON via `response_format={"type":"json_object"}` so
  free-form prose embedded in retrieved text can't escape the schema

If you operate in a high-sensitivity setting, disable literature retrieval
or restrict the OpenAlex search to a curated allow-list of paper IDs.

## Azure AD auth

- We use `DefaultAzureCredential` → `get_bearer_token_provider` with scope
  `https://cognitiveservices.azure.com/.default`.
- Tokens are acquired per-request by the OpenAI SDK; we never log them.
- The runtime needs the **Cognitive Services OpenAI User** role on the
  resource. Anything less will see `401` on first call.
- No API keys live in `.env`, nor in `RunManifest` events.

## Recommended hardening before exposing

1. Put the FastAPI app behind an authenticated reverse proxy (Cloudflare
   Access, oauth2-proxy, AAD App Proxy).
2. Switch the Stage II sandbox to Docker — drop `--network=none`,
   `--cap-drop=ALL`, `--security-opt=no-new-privileges`, read-only root
   filesystem, tmpfs `/tmp`.
3. Persist nothing user-supplied beyond `data/users/<id>/` and `runs/`.
   Both directories are local-only by default.
4. Rate-limit `POST /api/runs` and `POST /api/intent` (intent costs LLM
   tokens; runs cost LLM tokens + CPU).

## Reporting a vulnerability

Please email the maintainer listed in `pyproject.toml` instead of opening
a public issue.
