# NanoResearch UI

Minimal Vite + React + TypeScript + Tailwind v4 + `@assistant-ui/react` scaffold.

## Run it

```bash
# Terminal 1 — backend
cd ..
source .venv/bin/activate
nanoresearch serve            # FastAPI on :8000

# Terminal 2 — UI
cd ui
npm install
npm run dev                   # Vite on :5173
```

Open <http://localhost:5173>.

## Layout

- **Left column** — `UserPanel` (profile create/edit) + `RunPanel` (start a run).
- **Center** — `EventsList` (live SSE feed of trajectory events).
- **Right** — `AssistantThread` (assistant-ui chat that routes user messages to the run's feedback endpoint).

## Files

- `src/lib/api.ts` — typed fetch wrappers for `/api/*`
- `src/hooks/useRunStream.ts` — EventSource hook for `/api/runs/:id/stream`
- `src/components/*` — UI panels

The proxy in `vite.config.ts` forwards `/api/*` to the FastAPI server on :8000 so there's no CORS friction in dev.
