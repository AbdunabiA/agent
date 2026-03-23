"""Memory and workspace CLI commands."""

from __future__ import annotations

import asyncio
from pathlib import Path

import typer
import yaml
from rich.table import Table

from agent.cli._helpers import _load_config, console, err_console
from agent.config import AgentConfig
from agent.workspaces.manager import WorkspaceManager, WorkspaceNotFoundError

memory_app = typer.Typer(help="Memory management commands.")
workspace_app = typer.Typer(help="Workspace management commands.")


# --- Memory subcommands ---


@memory_app.command("export")
def memory_export(
    output: str = typer.Argument("memory_export.json", help="Output file path"),
    format: str = typer.Option("json", "--format", "-f", help="Export format: json or markdown"),
    config: str | None = typer.Option(None, "--config", "-c", help="Path to config file"),
) -> None:
    """Export memory (facts + soul) to file."""
    cfg = _load_config(config)
    asyncio.run(_memory_export(cfg, output, format))


async def _memory_export(cfg: AgentConfig, output: str, fmt: str) -> None:
    from agent.memory.database import Database
    from agent.memory.export import MemoryExporter
    from agent.memory.soul import SoulLoader
    from agent.memory.store import FactStore

    fact_store = None
    database = None

    try:
        database = Database(cfg.memory.db_path)
        await database.connect()
        fact_store = FactStore(database)
    except Exception as e:
        console.print(f"[yellow]Warning: Could not open database: {e}[/yellow]")

    soul_loader = SoulLoader(cfg.memory.soul_path)

    exporter = MemoryExporter(fact_store=fact_store, soul_loader=soul_loader)

    if fmt == "markdown":
        await exporter.export_markdown(output)
    else:
        await exporter.export_json(output)

    console.print(f"[green]Memory exported to:[/green] {output}")

    if database:
        await database.close()


@memory_app.command("import")
def memory_import(
    input_file: str = typer.Argument(help="Input JSON file path"),
    merge: bool = typer.Option(True, "--merge/--replace", help="Merge or replace existing facts"),
    config: str | None = typer.Option(None, "--config", "-c", help="Path to config file"),
) -> None:
    """Import memory from a JSON export file."""
    cfg = _load_config(config)
    asyncio.run(_memory_import(cfg, input_file, merge))


async def _memory_import(cfg: AgentConfig, input_file: str, merge: bool) -> None:
    from agent.memory.database import Database
    from agent.memory.export import MemoryExporter
    from agent.memory.soul import SoulLoader
    from agent.memory.store import FactStore

    database = Database(cfg.memory.db_path)
    await database.connect()
    fact_store = FactStore(database)
    soul_loader = SoulLoader(cfg.memory.soul_path)

    exporter = MemoryExporter(fact_store=fact_store, soul_loader=soul_loader)
    stats = await exporter.import_json(input_file, merge=merge)

    mode = "merged" if merge else "replaced"
    console.print(f"[green]Memory imported ({mode}):[/green]")
    console.print(f"  Facts: {stats['facts_imported']}")
    console.print(f"  Soul updated: {stats['soul_updated']}")

    await database.close()


@memory_app.command("stats")
def memory_stats_cmd(
    config: str | None = typer.Option(None, "--config", "-c", help="Path to config file"),
) -> None:
    """Show memory system statistics."""
    cfg = _load_config(config)
    asyncio.run(_memory_stats(cfg))


async def _memory_stats(cfg: AgentConfig) -> None:
    from agent.memory.database import Database
    from agent.memory.soul import SoulLoader
    from agent.memory.store import FactStore

    facts_count = 0
    vectors_count = 0
    soul_loaded = False

    try:
        database = Database(cfg.memory.db_path)
        await database.connect()
        fact_store = FactStore(database)
        facts_count = await fact_store.count()
        await database.close()
    except Exception:
        pass

    try:
        from agent.memory.vectors import VectorStore

        vs = VectorStore(persist_dir=cfg.memory.markdown_dir + "chroma")
        await vs.initialize()
        vectors_count = await vs.count()
    except Exception:
        pass

    soul_loader = SoulLoader(cfg.memory.soul_path)
    soul_loaded = bool(soul_loader.content)

    table = Table(title="Memory Stats", show_header=False)
    table.add_column("Key", style="cyan")
    table.add_column("Value")
    table.add_row("Facts", str(facts_count))
    table.add_row("Vectors", str(vectors_count))
    table.add_row("Soul loaded", "[green]yes[/green]" if soul_loaded else "[yellow]no[/yellow]")
    console.print(table)


# --- Workspace subcommands ---


@workspace_app.command("list")
def workspace_list(
    config: str | None = typer.Option(None, "--config", "-c", help="Path to config file"),
) -> None:
    """List all workspaces."""
    cfg = _load_config(config)
    manager = WorkspaceManager(cfg)
    manager.ensure_default()

    names = manager.discover()
    if not names:
        console.print("[dim]No workspaces found.[/dim]")
        return

    table = Table(
        title=f"Workspaces ({len(names)} found)",
        show_header=True,
        header_style="bold cyan",
    )
    table.add_column("Name", style="green")
    table.add_column("Display Name")
    table.add_column("Description")
    table.add_column("Active")

    for name in names:
        try:
            ws = manager.resolve(name)
            active = "yes" if name == cfg.workspaces.default else ""
            table.add_row(ws.name, ws.display_name, ws.description, active)
        except Exception:
            table.add_row(name, "(error)", "", "")

    console.print(table)


@workspace_app.command("create")
def workspace_create(
    name: str = typer.Argument(help="Workspace name (lowercase, no spaces)"),
    display_name: str = typer.Option("", "--display-name", "-d", help="Human-readable name"),
    description: str = typer.Option("", "--description", help="Workspace description"),
    clone: str | None = typer.Option(None, "--clone", help="Clone config from existing workspace"),
    config: str | None = typer.Option(None, "--config", "-c", help="Path to config file"),
) -> None:
    """Create a new workspace."""
    cfg = _load_config(config)
    manager = WorkspaceManager(cfg)

    try:
        ws = manager.create(
            name,
            display_name=display_name,
            description=description,
            clone_from=clone,
        )
        console.print(f"[green]Workspace created:[/green] {ws.name}")
        console.print(f"  Path: {ws.root_dir}")
        console.print(f"  Soul: {ws.soul_path}")
        console.print(f"  Data: {ws.data_dir}")
        if clone:
            console.print(f"  Cloned from: {clone}")
    except Exception as e:
        err_console.print(f"[red]Error:[/red] {e}")
        raise typer.Exit(1) from e


@workspace_app.command("info")
def workspace_info(
    name: str = typer.Argument(help="Workspace name"),
    config: str | None = typer.Option(None, "--config", "-c", help="Path to config file"),
) -> None:
    """Show detailed information about a workspace."""
    cfg = _load_config(config)
    manager = WorkspaceManager(cfg)

    try:
        ws = manager.resolve(name)
    except WorkspaceNotFoundError:
        err_console.print(f"[red]Workspace '{name}' not found.[/red]")
        raise typer.Exit(1) from None

    wc = ws.config

    # Basic info
    table = Table(
        title=f"Workspace: {ws.name} ({ws.display_name})",
        show_header=False,
    )
    table.add_column("Key", style="cyan")
    table.add_column("Value")
    table.add_row("Description", ws.description or "(none)")
    table.add_row("Path", str(ws.root_dir))

    # Overrides
    if wc.default_model is not None:
        table.add_row("Model", f"{wc.default_model} (override)")
    if wc.fallback_model is not None:
        table.add_row("Fallback Model", f"{wc.fallback_model} (override)")
    if wc.max_iterations is not None:
        table.add_row("Max Iterations", f"{wc.max_iterations} (override)")
    if wc.heartbeat_interval is not None:
        table.add_row("Heartbeat", f"{wc.heartbeat_interval} (override)")
    if wc.enabled_skills is not None:
        table.add_row("Enabled Skills", ", ".join(wc.enabled_skills))
    if wc.disabled_skills is not None:
        table.add_row("Disabled Skills", ", ".join(wc.disabled_skills))
    if wc.enabled_tools is not None:
        table.add_row("Enabled Tools", ", ".join(wc.enabled_tools))
    if wc.disabled_tools is not None:
        table.add_row("Disabled Tools", ", ".join(wc.disabled_tools))

    # Data stats
    db_path = Path(ws.get_db_path())
    if db_path.exists():
        size_kb = db_path.stat().st_size / 1024
        table.add_row("Database", f"{size_kb:.1f} KB")
    else:
        table.add_row("Database", "(not created)")

    soul_exists = ws.soul_path.exists()
    if soul_exists:
        soul_size = ws.soul_path.stat().st_size
        table.add_row("Soul", f"{soul_size} bytes")
    else:
        table.add_row("Soul", "(not created)")

    console.print(table)


@workspace_app.command("switch")
def workspace_switch(
    name: str = typer.Argument(help="Workspace name to switch to"),
    config: str | None = typer.Option(None, "--config", "-c", help="Path to config file"),
) -> None:
    """Set the active workspace (updates agent.yaml)."""
    cfg = _load_config(config)
    manager = WorkspaceManager(cfg)

    try:
        manager.resolve(name)
    except WorkspaceNotFoundError:
        err_console.print(f"[red]Workspace '{name}' not found.[/red]")
        raise typer.Exit(1) from None

    # Update the config file
    config_file = Path("agent.yaml")
    if config_file.exists():
        with open(config_file) as f:
            raw = yaml.safe_load(f) or {}
        raw.setdefault("workspaces", {})["default"] = name
        with open(config_file, "w") as f:
            yaml.dump(raw, f, default_flow_style=False, sort_keys=False)
        console.print(f"[green]Switched to workspace:[/green] {name}")
        console.print("[dim]Restart the agent for changes to take effect.[/dim]")
    else:
        console.print(f"[yellow]No agent.yaml found. Use --workspace {name} flag instead.[/yellow]")


@workspace_app.command("delete")
def workspace_delete(
    name: str = typer.Argument(help="Workspace name to delete"),
    confirm: bool = typer.Option(False, "--confirm", help="Confirm deletion (required)"),
    config: str | None = typer.Option(None, "--config", "-c", help="Path to config file"),
) -> None:
    """Delete a workspace and all its data."""
    cfg = _load_config(config)
    manager = WorkspaceManager(cfg)

    try:
        deleted = manager.delete(name, confirm=confirm)
        if deleted:
            console.print(f"[green]Workspace '{name}' deleted.[/green]")
        else:
            console.print(f"[yellow]Workspace '{name}' not found.[/yellow]")
    except ValueError as e:
        err_console.print(f"[red]Error:[/red] {e}")
        raise typer.Exit(1) from e


@workspace_app.command("current")
def workspace_current(
    config: str | None = typer.Option(None, "--config", "-c", help="Path to config file"),
) -> None:
    """Show the currently active workspace."""
    cfg = _load_config(config)
    console.print(f"Active workspace: [green]{cfg.workspaces.default}[/green]")
