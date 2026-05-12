# Contributing to NanoResearch

Thanks for your interest! NanoResearch is a research-grade implementation of
[arXiv:2605.10813](https://arxiv.org/abs/2605.10813). Contributions are welcome —
especially:

- 🧪 Additional benchmark tasks (the paper uses 20)
- 🌍 Field-specific prompt tuning (chemistry, medicine, social sciences, …)
- 🐛 Bug reports with reproductions
- 🧰 Docker / nsjail sandbox upgrade for Stage II
- 🧠 SDPO training improvements (MLX port, better LR schedules)

## Quick dev setup

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
pre-commit install

# Frontend
cd ui && npm install
```

## Test before opening a PR

```bash
# Backend
pytest -m "not azure and not local_model"   # offline; should be green

# Frontend
cd ui && npm run typecheck && npm run build
```

## Code style

- **Python**: ruff + black + mypy (config in `pyproject.toml`)
- **TypeScript**: `tsc -b --noEmit` + Vite build must pass
- Prefer immutable data; use pydantic v2 for any structured payload that
  crosses a process boundary.
- Match the existing module layout; **one stage = one module** under
  `src/nanoresearch/agents/`.

## Commit / PR conventions

- Use **conventional commits**: `feat:`, `fix:`, `refactor:`, `docs:`, `test:`,
  `chore:`, `perf:`, `ci:`.
- One logical change per PR. If you're adding a stage, include unit tests in
  the same PR.
- For UI changes, attach a screenshot or short screen recording.

## Reporting security issues

If you find a security issue (sandbox escape, prompt injection vector,
auth bypass), please **do not open a public issue**. Email the maintainer
listed in [`pyproject.toml`](pyproject.toml) instead.
