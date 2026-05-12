"""NanoResearch command-line interface (Phase 0: just env+backend checks)."""

from __future__ import annotations

import typer
from rich.console import Console
from rich.table import Table

from ..config import get_settings
from ..llm import AgentRole, AzureFoundryClient, ChatMessage, LLMRouter, Role
from ..logging import configure_logging, get_logger

app = typer.Typer(help="NanoResearch CLI", no_args_is_help=True)
console = Console()
_log = get_logger("cli")


@app.callback()
def _main(log_level: str = typer.Option("INFO", help="Logging level.")) -> None:
    configure_logging(log_level)


@app.command()
def settings() -> None:
    """Print the resolved Settings (sanity-check your .env)."""
    s = get_settings()
    table = Table(title="NanoResearch settings", show_header=True)
    table.add_column("key")
    table.add_column("value", overflow="fold")
    for k, v in s.model_dump().items():
        table.add_row(k, str(v))
    console.print(table)


@app.command()
def health(
    azure: bool = typer.Option(True, help="Probe the Azure GPT-5.1 deployment."),
    local: bool = typer.Option(False, help="Load Qwen2.5-7B locally and probe (heavy)."),
) -> None:
    """Validate that every configured backend is reachable."""
    ok = True
    if azure:
        ok &= _check_azure()
    if local:
        ok &= _check_local()
    if not ok:
        raise typer.Exit(code=1)
    console.print("[green]All checks passed.[/green]")


def _check_azure() -> bool:
    console.print("[bold]→ Azure AI Foundry (GPT-5.1) via AAD[/bold]")
    try:
        client = AzureFoundryClient()
        res = client.complete(
            [
                ChatMessage(Role.SYSTEM, "You are a terse echo. Reply with exactly: OK"),
                ChatMessage(Role.USER, "Reply OK"),
            ],
            max_tokens=8,
            temperature=0.0,
        )
        console.print(
            f"  [green]✓[/green] reply={res.text!r}  "
            f"tokens={res.prompt_tokens}+{res.completion_tokens}  "
            f"{res.latency_ms:.0f}ms"
        )
        return True
    except Exception as e:  # noqa: BLE001 - we want the full error message
        console.print(f"  [red]✗ {type(e).__name__}: {e}[/red]")
        console.print(
            "    Hint: run `az login` and ensure your account has the "
            "[bold]Cognitive Services OpenAI User[/bold] role on the resource."
        )
        return False


def _check_local() -> bool:
    console.print("[bold]→ Local Qwen2.5-7B (MPS)[/bold]")
    try:
        router = LLMRouter()
        res = router.complete(
            AgentRole.PLANNER,
            [
                ChatMessage(Role.USER, "Reply with exactly: OK"),
            ],
            max_tokens=8,
            temperature=0.0,
        )
        console.print(
            f"  [green]✓[/green] reply={res.text!r}  "
            f"tokens={res.prompt_tokens}+{res.completion_tokens}  "
            f"{res.latency_ms:.0f}ms"
        )
        return True
    except Exception as e:  # noqa: BLE001
        console.print(f"  [red]✗ {type(e).__name__}: {e}[/red]")
        console.print(
            "    Hint: install local deps with `pip install -e .[local]` "
            "and ensure Qwen2.5-7B-Instruct is downloaded."
        )
        return False


@app.command()
def serve(
    host: str = typer.Option("127.0.0.1", help="Bind host."),
    port: int = typer.Option(8000, help="Bind port."),
    reload: bool = typer.Option(False, help="Auto-reload on code changes (dev only)."),
    access_log: bool = typer.Option(
        False, help="Log every HTTP request (noisy with SSE + polling)."
    ),
) -> None:
    """Run the FastAPI server for the UI."""
    import uvicorn

    uvicorn.run(
        "nanoresearch.api.app:app",
        host=host,
        port=port,
        reload=reload,
        log_level="info",
        access_log=access_log,
    )


if __name__ == "__main__":
    app()
