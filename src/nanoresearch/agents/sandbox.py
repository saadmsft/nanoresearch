"""Subprocess sandbox for Stage II execution.

Each ``run()`` writes the project files into a per-run workspace directory,
launches the entrypoint with strict limits, and captures truncated
stdout/stderr. We intentionally avoid Docker for portability — the user's
plan committed to subprocess-first.
"""

from __future__ import annotations

import os
import resource
import shutil
import subprocess
import sys
import time
from pathlib import Path

from ..logging import get_logger
from .artefacts import CodeFile, ExecutionResult

_log = get_logger(__name__)


# ----------------------------------------------------------------- limits


_MEMORY_LIMIT_MB = 2048      # per-process RAM cap
_DEFAULT_TIMEOUT = 300       # seconds (5 min)
_STDOUT_TAIL_BYTES = 8000
_STDERR_TAIL_BYTES = 4000


def _preexec_limits():
    """Apply RLIMITs in the child process."""
    # Address space (Linux respects RLIMIT_AS; macOS may ignore but caps mmap).
    soft = _MEMORY_LIMIT_MB * 1024 * 1024
    try:
        resource.setrlimit(resource.RLIMIT_AS, (soft, soft))
    except (ValueError, OSError):
        pass
    # CPU time
    try:
        resource.setrlimit(resource.RLIMIT_CPU, (_DEFAULT_TIMEOUT + 30, _DEFAULT_TIMEOUT + 60))
    except (ValueError, OSError):
        pass


def _tail(data: bytes, n: int) -> str:
    text = data.decode("utf-8", errors="replace")
    if len(text) <= n:
        return text
    return "…[truncated]…\n" + text[-n:]


# ----------------------------------------------------------------- workspace


def write_files(workspace: Path, files: list[CodeFile]) -> None:
    workspace.mkdir(parents=True, exist_ok=True)
    for f in files:
        target = (workspace / f.path).resolve()
        # Guard against path traversal — written files must live under workspace.
        if not str(target).startswith(str(workspace.resolve())):
            raise ValueError(f"Refusing to write outside workspace: {f.path}")
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(f.content, encoding="utf-8")


def list_produced(workspace: Path, before_mtime: float) -> list[str]:
    out: list[str] = []
    for p in workspace.rglob("*"):
        if p.is_file() and p.stat().st_mtime > before_mtime:
            out.append(str(p.relative_to(workspace)))
    return sorted(out)


# ----------------------------------------------------------------- run


def run_sandboxed(
    *,
    workspace: Path,
    entrypoint: str,
    timeout_seconds: int = _DEFAULT_TIMEOUT,
    extra_env: dict[str, str] | None = None,
) -> ExecutionResult:
    """Run ``python <entrypoint>`` inside ``workspace``."""
    workspace = workspace.resolve()
    workspace.mkdir(parents=True, exist_ok=True)
    before = time.time()

    env = os.environ.copy()
    # Sterilise the child environment.
    env.pop("PYTHONSTARTUP", None)
    env["PYTHONUNBUFFERED"] = "1"
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env["NO_PROXY"] = "*"
    # Block obvious network access by yanking proxy hints.
    for k in ("HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy"):
        env.pop(k, None)
    if extra_env:
        env.update(extra_env)

    cmd = [sys.executable, entrypoint]
    _log.info("sandbox_run", workspace=str(workspace), cmd=cmd, timeout_s=timeout_seconds)

    start = time.time()
    timed_out = False
    try:
        completed = subprocess.run(
            cmd,
            cwd=str(workspace),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=timeout_seconds,
            preexec_fn=_preexec_limits if os.name != "nt" else None,
            check=False,
        )
        exit_code = completed.returncode
        stdout_b = completed.stdout
        stderr_b = completed.stderr
    except subprocess.TimeoutExpired as e:
        timed_out = True
        exit_code = None
        stdout_b = e.stdout or b""
        stderr_b = (e.stderr or b"") + f"\n[timeout after {timeout_seconds}s]".encode()
    except FileNotFoundError as e:
        return ExecutionResult(
            success=False,
            exit_code=None,
            duration_seconds=0.0,
            stdout_tail="",
            stderr_tail=f"entrypoint not found: {e}",
            timed_out=False,
            workspace_path=str(workspace),
        )

    duration = time.time() - start
    success = exit_code == 0 and not timed_out

    return ExecutionResult(
        success=success,
        exit_code=exit_code,
        duration_seconds=duration,
        stdout_tail=_tail(stdout_b, _STDOUT_TAIL_BYTES),
        stderr_tail=_tail(stderr_b, _STDERR_TAIL_BYTES),
        timed_out=timed_out,
        workspace_path=str(workspace),
        produced_files=list_produced(workspace, before),
    )


# ----------------------------------------------------------------- cleanup


def reset_workspace(workspace: Path) -> None:
    if workspace.exists():
        shutil.rmtree(workspace)
    workspace.mkdir(parents=True, exist_ok=True)
