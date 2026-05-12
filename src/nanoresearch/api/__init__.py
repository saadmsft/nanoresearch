"""HTTP API exposing NanoResearch to the React UI.

Endpoints (v1):

- ``POST   /api/users``                 — upsert a user profile.
- ``GET    /api/users``                 — list profiles.
- ``GET    /api/users/{user_id}``       — fetch one profile.
- ``GET    /api/users/{user_id}/skills``      — Skill Bank contents.
- ``GET    /api/users/{user_id}/memories``    — Memory Module contents.
- ``POST   /api/users/{user_id}/train``       — trigger SDPO round.
- ``POST   /api/runs``                  — start a new run (topic + user_id).
- ``GET    /api/runs``                  — list runs in this process.
- ``GET    /api/runs/{run_id}``         — current snapshot.
- ``GET    /api/runs/{run_id}/stream``  — Server-Sent Events stream of events.
- ``POST   /api/runs/{run_id}/feedback``      — submit free-form feedback.

For the v1 scaffold we keep state in-process so a single server pid sees every
run. This is good enough for local research workflows; a Redis/Postgres
backend can be plugged in via the same routes later.
"""

from .app import create_app

__all__ = ["create_app"]
