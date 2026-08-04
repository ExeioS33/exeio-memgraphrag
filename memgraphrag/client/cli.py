"""Typer + Rich CLI for MemGraphRAG (`memgraphrag-cli`)."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Optional

import typer
from rich import box
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table

from memgraphrag.client.env import load_client_env
from memgraphrag.client.http import MemGraphRAGClient
from memgraphrag.client.optimize import expand_grid, run_optimize
from memgraphrag.client.params import (
    PRESETS,
    QUERY_PARAMS,
    clean_params,
    default_sweep_grid,
)
from memgraphrag.utils.http_ssl import describe_ssl_verify, reset_ssl_verify_cache

load_client_env()
reset_ssl_verify_cache()

app = typer.Typer(
    name="memgraphrag-cli",
    help="🎮 Lightweight CLI for the MemGraphRAG API container.",
    no_args_is_help=True,
    rich_markup_mode="rich",
)
docs_app = typer.Typer(help="📥 Document ingest & status commands.", no_args_is_help=True)
graph_app = typer.Typer(help="🕸️ Graph exploration commands.", no_args_is_help=True)
app.add_typer(docs_app, name="docs")
app.add_typer(graph_app, name="graph")

console = Console()


def _client(
    server: Optional[str],
    api_key: Optional[str],
) -> MemGraphRAGClient:
    return MemGraphRAGClient(base_url=server, api_key=api_key)


def _print_json(data: Any) -> None:
    console.print_json(json.dumps(data, ensure_ascii=False, default=str))


def _err(exc: Exception) -> None:
    console.print(f"[bold red]💥 Error:[/] {exc}")
    raise typer.Exit(code=1) from exc


# --------------------------------------------------------------------------- #
# Global options helpers
# --------------------------------------------------------------------------- #
ServerOpt = typer.Option(
    None,
    "--server",
    "-s",
    envvar="MEMGRAPHRAG_SERVER_URL",
    help="API base URL (default http://localhost:9621)",
)
ApiKeyOpt = typer.Option(
    None,
    "--api-key",
    "-k",
    envvar="MEMGRAPHRAG_API_KEY",
    help="X-API-Key value",
)


@app.command("health")
def health_cmd(
    server: Optional[str] = ServerOpt,
    api_key: Optional[str] = ApiKeyOpt,
) -> None:
    """❤️  Ping /health and show versions + pipeline status."""
    try:
        with _client(server, api_key) as c:
            data = c.health()
    except Exception as exc:  # noqa: BLE001
        _err(exc)
    busy = data.get("pipeline_busy")
    badge = "🟡 BUSY indexing" if busy else "🟢 IDLE"
    console.print(
        Panel.fit(
            f"[bold]status[/] = {data.get('status')}\n"
            f"[bold]core[/]   = {data.get('core_version')}\n"
            f"[bold]api[/]    = {data.get('api_version')}\n"
            f"[bold]auth[/]   = {data.get('auth_mode')}\n"
            f"[bold]pipe[/]   = {badge}\n"
            f"[bold]dir[/]    = {data.get('working_dir')}",
            title="🏥 MemGraphRAG Health",
            border_style="green",
        )
    )


@app.command("params")
def params_cmd() -> None:
    """🗺️  Feature map — query params, presets, and API coverage."""
    table = Table(title="🎛️ Tunable query parameters", box=box.ROUNDED)
    table.add_column("Emoji", justify="center")
    table.add_column("Name")
    table.add_column("Kind")
    table.add_column("Default")
    table.add_column("Grid / choices")
    table.add_column("Help")
    for p in QUERY_PARAMS:
        grid = ", ".join(str(x) for x in (p.grid or p.choices or ()))
        table.add_row(
            p.emoji,
            p.name,
            p.kind,
            str(p.default),
            grid or "—",
            p.help,
        )
    console.print(table)

    preset_table = Table(title="✨ Presets", box=box.SIMPLE)
    preset_table.add_column("Name")
    preset_table.add_column("Params")
    for name, vals in PRESETS.items():
        preset_table.add_row(name, json.dumps(vals))
    console.print(preset_table)

    cover = Table(title="📡 Endpoint coverage", box=box.SIMPLE_HEAVY)
    cover.add_column("HTTP")
    cover.add_column("CLI")
    cover.add_column("UI tab")
    rows = [
        ("GET /health", "health", "🏠 Home"),
        ("POST /query", "query", "💬 Query"),
        ("POST /query/data", "query --data-only", "💬 Query (context)"),
        ("POST /query/stream", "query --stream", "💬 Query (stream)"),
        ("POST /documents/upload", "docs upload|upload-dir|upload-url", "📥 Ingest"),
        ("POST /documents/text", "docs text", "📥 Ingest"),
        ("GET /documents/", "docs list", "📥 Ingest"),
        ("POST /documents/scan", "docs scan", "📥 Ingest"),
        ("DELETE /documents/", "docs clear (stub)", "📥 Ingest"),
        ("GET /graphs", "graph show", "🕸️ Graph"),
        ("GET /graph/label/list", "graph labels", "🕸️ Graph"),
        ("(client-side)", "optimize", "🧪 Optimize"),
    ]
    for row in rows:
        cover.add_row(*row)
    console.print(cover)


@app.command("query")
def query_cmd(
    question: str = typer.Argument(..., help="Natural-language question"),
    mode: Optional[str] = typer.Option(None, help="ppr | naive | context | bypass"),
    top_k: Optional[int] = typer.Option(None, help="Passages to return"),
    linking_top_k: Optional[int] = typer.Option(None, help="Seed linking top-k"),
    passage_node_weight: Optional[float] = typer.Option(None, help="Passage seed weight"),
    damping: Optional[float] = typer.Option(None, help="PPR damping"),
    fact_similarity_threshold: Optional[float] = typer.Option(
        None, help="Fact similarity threshold"
    ),
    skip_fact_rerank: Optional[bool] = typer.Option(None, help="Skip fact rerank"),
    schema_top_k: Optional[int] = typer.Option(None, help="Ontology schema linking top-k"),
    schema_node_weight: Optional[float] = typer.Option(
        None, help="Schema-expanded seed weight"
    ),
    user_prompt: Optional[str] = typer.Option(None, help="Extra system prompt"),
    preset: Optional[str] = typer.Option(
        None, "--preset", help="Preset name (e.g. '⚖️ Balanced')"
    ),
    data_only: bool = typer.Option(False, "--data-only", help="Use /query/data"),
    stream: bool = typer.Option(False, "--stream", help="Use /query/stream SSE"),
    raw: bool = typer.Option(False, "--raw", help="Print raw JSON"),
    server: Optional[str] = ServerOpt,
    api_key: Optional[str] = ApiKeyOpt,
) -> None:
    """💬 Ask the MemGraphRAG server a question."""
    params: dict[str, Any] = {}
    if preset:
        if preset not in PRESETS:
            console.print(f"[red]Unknown preset:[/] {preset}. Choose from: {list(PRESETS)}")
            raise typer.Exit(1)
        params.update(PRESETS[preset])
    for key, val in {
        "mode": mode,
        "top_k": top_k,
        "linking_top_k": linking_top_k,
        "passage_node_weight": passage_node_weight,
        "damping": damping,
        "fact_similarity_threshold": fact_similarity_threshold,
        "skip_fact_rerank": skip_fact_rerank,
        "schema_top_k": schema_top_k,
        "schema_node_weight": schema_node_weight,
        "user_prompt": user_prompt,
    }.items():
        if val is not None:
            params[key] = val
    params = clean_params(params)

    try:
        with _client(server, api_key) as c:
            if stream:
                console.print("[cyan]📡 Streaming…[/]")
                chunks: list[str] = []
                for payload in c.query_stream(question, **params):
                    if payload == "[DONE]":
                        break
                    try:
                        obj = json.loads(payload)
                        text = obj.get("response") or obj.get("error") or payload
                    except json.JSONDecodeError:
                        text = payload
                    chunks.append(str(text))
                    console.print(text, end="")
                console.print()
                return
            with Progress(
                SpinnerColumn(), TextColumn("[progress.description]{task.description}")
            ) as progress:
                progress.add_task("🧠 Thinking…", total=None)
                if data_only:
                    data = c.query_data(question, **params)
                else:
                    data = c.query(question, **params)
    except Exception as exc:  # noqa: BLE001
        _err(exc)

    if raw:
        _print_json(data)
        return

    if data_only:
        payload = data.get("data") if isinstance(data.get("data"), dict) else data
        docs = (payload or {}).get("docs") or []
        scores = (payload or {}).get("doc_scores") or []
        console.print(Panel.fit("📚 Retrieved context", border_style="blue"))
        for i, doc in enumerate(docs):
            score = scores[i] if i < len(scores) else "?"
            console.print(f"[bold]#{i + 1}[/] score={score}\n{doc}\n")
        return

    answer = data.get("answer") or data.get("response") or ""
    console.print(Panel(str(answer), title="✨ Answer", border_style="magenta"))
    docs = data.get("docs") or []
    scores = data.get("doc_scores") or []
    if docs:
        table = Table(title="📎 Evidence", box=box.MINIMAL)
        table.add_column("#", justify="right")
        table.add_column("Score")
        table.add_column("Snippet")
        for i, doc in enumerate(docs):
            score = str(scores[i]) if i < len(scores) else "—"
            snippet = (doc[:160] + "…") if len(doc) > 160 else doc
            table.add_row(str(i + 1), score, snippet)
        console.print(table)


# --------------------------------------------------------------------------- #
# docs
# --------------------------------------------------------------------------- #
@docs_app.command("list")
def docs_list(
    server: Optional[str] = ServerOpt,
    api_key: Optional[str] = ApiKeyOpt,
    raw: bool = typer.Option(False, "--raw"),
) -> None:
    """📋 List document statuses."""
    try:
        with _client(server, api_key) as c:
            data = c.list_documents()
    except Exception as exc:  # noqa: BLE001
        _err(exc)
    if raw:
        _print_json(data)
        return
    statuses = data.get("statuses") or {}
    table = Table(title="📥 Documents", box=box.ROUNDED)
    table.add_column("Doc ID")
    table.add_column("Status")
    table.add_column("File")
    status_emoji = {
        "pending": "⏳",
        "parsing": "🔍",
        "processing": "⚙️",
        "processed": "✅",
        "failed": "❌",
    }
    if isinstance(statuses, dict):
        for doc_id, meta in statuses.items():
            if isinstance(meta, dict):
                st = str(meta.get("status") or meta.get("doc_status") or "?")
                path = str(meta.get("file_path") or meta.get("content_summary") or "")
            else:
                st, path = str(meta), ""
            emoji = status_emoji.get(st.lower(), "📄")
            table.add_row(str(doc_id), f"{emoji} {st}", path[:80])
    console.print(table)


@docs_app.command("upload")
def docs_upload(
    file: Path = typer.Argument(..., exists=True, dir_okay=False, readable=True),
    server: Optional[str] = ServerOpt,
    api_key: Optional[str] = ApiKeyOpt,
) -> None:
    """📤 Upload a single local file."""
    try:
        with _client(server, api_key) as c:
            with console.status(f"📤 Uploading {file.name}…"):
                data = c.upload_file(file)
    except Exception as exc:  # noqa: BLE001
        _err(exc)
    console.print(f"[green]✅ Queued[/] doc_id={data.get('doc_id')} → {data.get('filename')}")
    _print_json(data)


@docs_app.command("upload-dir")
def docs_upload_dir(
    directory: Path = typer.Argument(..., exists=True, file_okay=False),
    recursive: bool = typer.Option(True, "--recursive/--no-recursive"),
    server: Optional[str] = ServerOpt,
    api_key: Optional[str] = ApiKeyOpt,
) -> None:
    """📁 Upload supported files from a local directory."""
    try:
        with _client(server, api_key) as c:
            with console.status(f"📁 Walking {directory}…"):
                results = c.upload_directory(directory, recursive=recursive)
    except Exception as exc:  # noqa: BLE001
        _err(exc)
    console.print(f"[green]✅ Uploaded {len(results)} file(s)[/]")
    for r in results:
        console.print(f"  • {r.get('filename')} → {r.get('doc_id')}")


@docs_app.command("upload-url")
def docs_upload_url(
    url: str = typer.Argument(..., help="HTTP(S) URL to download + ingest"),
    filename: Optional[str] = typer.Option(None, "--filename", "-f"),
    server: Optional[str] = ServerOpt,
    api_key: Optional[str] = ApiKeyOpt,
) -> None:
    """🌐 Download a URL then upload it for indexing."""
    try:
        with _client(server, api_key) as c:
            with console.status("🌐 Downloading + uploading…"):
                data = c.upload_url(url, filename=filename)
    except Exception as exc:  # noqa: BLE001
        _err(exc)
    console.print(f"[green]✅ Queued[/] {data.get('filename')} → {data.get('doc_id')}")
    _print_json(data)


@docs_app.command("text")
def docs_text(
    text: str = typer.Argument(..., help="Raw text to index inline"),
    doc_id: Optional[str] = typer.Option(None, "--doc-id"),
    server: Optional[str] = ServerOpt,
    api_key: Optional[str] = ApiKeyOpt,
) -> None:
    """📝 Insert raw text (synchronous index)."""
    try:
        with _client(server, api_key) as c:
            with console.status("📝 Indexing text…"):
                data = c.insert_text(text, doc_id=doc_id)
    except Exception as exc:  # noqa: BLE001
        _err(exc)
    console.print(f"[green]✅ Indexed[/] doc_id={data.get('doc_id')}")
    _print_json(data)


@docs_app.command("scan")
def docs_scan(
    server: Optional[str] = ServerOpt,
    api_key: Optional[str] = ApiKeyOpt,
) -> None:
    """🔎 Rescan the server input directory and enqueue new files."""
    try:
        with _client(server, api_key) as c:
            data = c.scan_input_dir()
    except Exception as exc:  # noqa: BLE001
        _err(exc)
    console.print(
        f"[green]✅ Scan queued[/] files_found={data.get('files_found')} "
        f"dir={data.get('input_dir')}"
    )


@docs_app.command("clear")
def docs_clear(
    server: Optional[str] = ServerOpt,
    api_key: Optional[str] = ApiKeyOpt,
    yes: bool = typer.Option(False, "--yes", "-y", help="Skip confirmation"),
) -> None:
    """🗑️  Call DELETE /documents/ (POC stub — may be not_implemented)."""
    if not yes and not typer.confirm("Really call DELETE /documents/?"):
        raise typer.Abort()
    try:
        with _client(server, api_key) as c:
            data = c.clear_documents()
    except Exception as exc:  # noqa: BLE001
        _err(exc)
    console.print(f"[yellow]⚠️  Server says:[/] {data}")


# --------------------------------------------------------------------------- #
# graph
# --------------------------------------------------------------------------- #
@graph_app.command("labels")
def graph_labels(
    server: Optional[str] = ServerOpt,
    api_key: Optional[str] = ApiKeyOpt,
) -> None:
    """🏷️  List graph labels / layers."""
    try:
        with _client(server, api_key) as c:
            data = c.list_labels()
    except Exception as exc:  # noqa: BLE001
        _err(exc)
    labels = data.get("labels") or []
    console.print("🏷️  " + (", ".join(labels) if labels else "(none)"))


@graph_app.command("show")
def graph_show(
    label: Optional[str] = typer.Option(None, "--label", "-l"),
    limit: int = typer.Option(50, "--limit", "-n", min=1, max=5000),
    server: Optional[str] = ServerOpt,
    api_key: Optional[str] = ApiKeyOpt,
    raw: bool = typer.Option(False, "--raw"),
) -> None:
    """🕸️  Explore nodes and edges."""
    try:
        with _client(server, api_key) as c:
            data = c.explore_graph(label=label, limit=limit)
    except Exception as exc:  # noqa: BLE001
        _err(exc)
    if raw:
        _print_json(data)
        return
    nodes = data.get("nodes") or []
    edges = data.get("edges") or []
    console.print(f"[bold]Nodes[/]={len(nodes)}  [bold]Edges[/]={len(edges)}")
    ntable = Table(title="Nodes (sample)", box=box.MINIMAL)
    ntable.add_column("id")
    ntable.add_column("label/layer")
    for n in nodes[:20]:
        nid = str(n.get("id") or n.get("node_id") or "")
        lab = str(n.get("label") or n.get("layer") or n.get("entity_type") or "")
        ntable.add_row(nid[:40], lab)
    console.print(ntable)


# --------------------------------------------------------------------------- #
# optimize
# --------------------------------------------------------------------------- #
@app.command("optimize")
def optimize_cmd(
    question: str = typer.Argument(..., help="Question used for the sweep"),
    extra_question: Optional[list[str]] = typer.Option(
        None, "--extra-question", "-q", help="Additional questions for phase-1"
    ),
    mode: Optional[str] = typer.Option(
        None, help="Comma-separated modes to sweep (default from grid)"
    ),
    top_k: Optional[str] = typer.Option(None, help="Comma-separated ints"),
    linking_top_k: Optional[str] = typer.Option(None, help="Comma-separated ints"),
    passage_node_weight: Optional[str] = typer.Option(
        None, help="Comma-separated floats"
    ),
    damping: Optional[str] = typer.Option(None, help="Comma-separated floats"),
    fact_similarity_threshold: Optional[str] = typer.Option(
        None, help="Comma-separated floats"
    ),
    skip_fact_rerank: Optional[str] = typer.Option(
        None, help="Comma-separated bools, e.g. false,true"
    ),
    schema_top_k: Optional[str] = typer.Option(None, help="Comma-separated ints"),
    schema_node_weight: Optional[str] = typer.Option(
        None, help="Comma-separated floats"
    ),
    top_n: int = typer.Option(3, "--top-n", help="How many winners to LLM-judge"),
    judge: bool = typer.Option(True, "--judge/--no-judge"),
    output: Optional[Path] = typer.Option(None, "--output", "-o", help="Write JSON report"),
    server: Optional[str] = ServerOpt,
    api_key: Optional[str] = ApiKeyOpt,
) -> None:
    """🧪 Hybrid param sweep (retrieval metrics + optional LLM judge)."""

    def _parse_list(raw: Optional[str], cast):
        if raw is None:
            return None
        parts = [p.strip() for p in raw.split(",") if p.strip()]
        out = []
        for p in parts:
            if cast is bool:
                out.append(p.lower() in {"1", "true", "yes", "y"})
            else:
                out.append(cast(p))
        return out

    grid = default_sweep_grid()
    overrides = {
        "mode": _parse_list(mode, str),
        "top_k": _parse_list(top_k, int),
        "linking_top_k": _parse_list(linking_top_k, int),
        "passage_node_weight": _parse_list(passage_node_weight, float),
        "damping": _parse_list(damping, float),
        "fact_similarity_threshold": _parse_list(fact_similarity_threshold, float),
        "skip_fact_rerank": _parse_list(skip_fact_rerank, bool),
        "schema_top_k": _parse_list(schema_top_k, int),
        "schema_node_weight": _parse_list(schema_node_weight, float),
    }
    for key, vals in overrides.items():
        if vals is not None:
            grid[key] = vals

    def on_progress(phase: str, i: int, total: int) -> None:
        console.print(f"  [{phase}] {i}/{total}", highlight=False)

    try:
        with _client(server, api_key) as c:
            console.print(f"🧪 Sweeping [bold]{len(expand_grid(grid))}[/] combos…")
            report = run_optimize(
                c,
                question,
                grid=grid,
                questions=extra_question,
                top_n=top_n,
                judge=judge,
                progress=on_progress,
            )
    except Exception as exc:  # noqa: BLE001
        _err(exc)

    table = Table(title="🏆 Leaderboard", box=box.ROUNDED)
    table.add_column("#", justify="right")
    table.add_column("Final")
    table.add_column("Retrieval")
    table.add_column("Judge")
    table.add_column("Params")
    for idx, row in enumerate(report.results[:15], start=1):
        table.add_row(
            str(idx),
            f"{row.final_score:.3f}",
            f"{row.retrieval_score:.3f}",
            "—" if row.judge_score is None else f"{row.judge_score:.1f}",
            json.dumps(row.params, ensure_ascii=False),
        )
    console.print(table)
    console.print(
        Panel(
            json.dumps(report.recommended, indent=2, ensure_ascii=False),
            title="✨ Recommended params",
            border_style="green",
        )
    )
    if output:
        output.write_text(report.to_json(), encoding="utf-8")
        console.print(f"[green]💾 Wrote[/] {output}")


def main() -> None:
    """Console-script entry point."""
    try:
        app()
    except KeyboardInterrupt:
        console.print("\n[yellow]Interrupted[/]")
        sys.exit(130)


if __name__ == "__main__":
    main()
