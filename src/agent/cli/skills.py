"""Skill management CLI commands."""

from __future__ import annotations

import asyncio
import contextlib
from pathlib import Path

import typer
from rich.table import Table

from agent.cli._helpers import _load_config, console

skills_app = typer.Typer(help="Skill management commands.")


@skills_app.command("list")
def skills_list(
    config: str | None = typer.Option(None, "--config", "-c", help="Path to config file"),
) -> None:
    """List all discovered skills."""
    cfg = _load_config(config)
    asyncio.run(_skills_list(cfg))


async def _skills_list(cfg: object) -> None:
    from agent.core.events import EventBus
    from agent.core.scheduler import TaskScheduler
    from agent.skills.manager import SkillManager
    from agent.tools.registry import ToolRegistry

    tool_registry = ToolRegistry()
    event_bus = EventBus()
    scheduler = TaskScheduler(event_bus)
    manager = SkillManager(
        config=cfg.skills,
        tool_registry=tool_registry,
        event_bus=event_bus,
        scheduler=scheduler,
    )

    with contextlib.suppress(Exception):
        await manager.discover_and_load()

    skills = manager.list_skills()
    if not skills:
        console.print("[dim]No skills found.[/dim]")
        return

    table = Table(title="Skills", show_header=True, header_style="bold cyan")
    table.add_column("Name", style="green")
    table.add_column("Version")
    table.add_column("Status")
    table.add_column("Tools")
    table.add_column("Permissions")

    for s in skills:
        status = "[green]loaded[/green]" if s["loaded"] else "[yellow]not loaded[/yellow]"
        tools = ", ".join(s["tools"]) if s["tools"] else "-"
        perms = ", ".join(s["permissions"])
        table.add_row(
            s["display_name"] or s["name"],
            s["version"],
            status,
            tools,
            perms,
        )

    console.print(table)


@skills_app.command("info")
def skills_info(
    name: str = typer.Argument(help="Skill name"),
    config: str | None = typer.Option(None, "--config", "-c", help="Path to config file"),
) -> None:
    """Show detailed information about a skill."""
    cfg = _load_config(config)
    asyncio.run(_skills_info(cfg, name))


async def _skills_info(cfg: object, name: str) -> None:
    from agent.core.events import EventBus
    from agent.core.scheduler import TaskScheduler
    from agent.skills.manager import SkillManager
    from agent.tools.registry import ToolRegistry

    tool_registry = ToolRegistry()
    event_bus = EventBus()
    scheduler = TaskScheduler(event_bus)
    manager = SkillManager(
        config=cfg.skills,
        tool_registry=tool_registry,
        event_bus=event_bus,
        scheduler=scheduler,
    )

    with contextlib.suppress(Exception):
        await manager.discover_and_load()

    skills = manager.list_skills()
    skill = next((s for s in skills if s["name"] == name), None)
    if not skill:
        console.print(f"[red]Skill '{name}' not found.[/red]")
        raise typer.Exit(1)

    table = Table(show_header=False, title=skill["display_name"] or skill["name"])
    table.add_column("Key", style="cyan")
    table.add_column("Value")
    table.add_row("Name", skill["name"])
    table.add_row("Version", skill["version"])
    table.add_row("Author", skill.get("author", "") or "-")
    table.add_row("Description", skill.get("description", "") or "-")
    table.add_row("Status", "loaded" if skill["loaded"] else "not loaded")
    table.add_row("Permissions", ", ".join(skill["permissions"]))
    table.add_row("Tools", ", ".join(skill["tools"]) if skill["tools"] else "-")
    table.add_row("Path", skill.get("path", ""))
    console.print(table)


@skills_app.command("enable")
def skills_enable(
    name: str = typer.Argument(help="Skill name to enable"),
    config: str | None = typer.Option(None, "--config", "-c", help="Path to config file"),
) -> None:
    """Enable a disabled skill (removes from disabled list in config)."""
    from agent.config import _save_config_to_disk

    cfg = _load_config(config)
    if name in cfg.skills.disabled:
        cfg.skills.disabled.remove(name)
        _save_config_to_disk(cfg)
        console.print(f"[green]Skill '{name}' enabled and saved to config.[/green]")
    else:
        console.print(f"[dim]Skill '{name}' is already enabled.[/dim]")


@skills_app.command("disable")
def skills_disable(
    name: str = typer.Argument(help="Skill name to disable"),
    config: str | None = typer.Option(None, "--config", "-c", help="Path to config file"),
) -> None:
    """Disable a skill (adds to disabled list in config)."""
    from agent.config import _save_config_to_disk

    cfg = _load_config(config)
    if name not in cfg.skills.disabled:
        cfg.skills.disabled.append(name)
        _save_config_to_disk(cfg)
        console.print(f"[yellow]Skill '{name}' disabled and saved to config.[/yellow]")
    else:
        console.print(f"[dim]Skill '{name}' is already disabled.[/dim]")


@skills_app.command("reload")
def skills_reload(
    name: str = typer.Argument(help="Skill name to reload"),
    config: str | None = typer.Option(None, "--config", "-c", help="Path to config file"),
) -> None:
    """Reload a skill (hot-reload)."""
    console.print(
        "[dim]Skill reload is only available while the agent is running "
        "(agent start). Use the API: POST /api/v1/skills/{name}/reload[/dim]"
    )


@skills_app.command("create")
def skills_create(
    name: str = typer.Argument(help="Skill name (e.g. my-skill)"),
    config: str | None = typer.Option(None, "--config", "-c", help="Path to config file"),
) -> None:
    """Scaffold a new skill directory."""
    cfg = _load_config(config)
    skills_dir = Path(cfg.skills.directory)
    skill_path = skills_dir / name

    if skill_path.exists():
        console.print(f"[red]Directory already exists:[/red] {skill_path}")
        raise typer.Exit(1)

    skill_path.mkdir(parents=True)

    # SKILL.md
    (skill_path / "SKILL.md").write_text(
        f"---\n"
        f"name: {name}\n"
        f"description: A new skill\n"
        f"version: '0.1.0'\n"
        f"permissions:\n"
        f"  - safe\n"
        f"---\n\n"
        f"# {name.replace('-', ' ').title()}\n\n"
        f"Describe your skill here.\n",
        encoding="utf-8",
    )

    # main.py
    class_name = name.replace("-", " ").replace("_", " ").title().replace(" ", "") + "Skill"
    (skill_path / "main.py").write_text(
        f"from agent.skills.base import Skill\n\n\n"
        f"class {class_name}(Skill):\n"
        f"    async def setup(self) -> None:\n"
        f"        self.register_tool(\n"
        f'            name="hello",\n'
        f'            description="Say hello",\n'
        f"            function=self._hello,\n"
        f'            tier="safe",\n'
        f"        )\n\n"
        f'    async def _hello(self, name: str = "world") -> str:\n'
        f'        return f"Hello, {{name}}!"\n',
        encoding="utf-8",
    )

    console.print(f"[green]Skill scaffolded:[/green] {skill_path}")
    console.print(f"  - {skill_path / 'SKILL.md'}")
    console.print(f"  - {skill_path / 'main.py'}")


@skills_app.command("staged")
def skills_staged(
    config: str | None = typer.Option(None, "--config", "-c", help="Path to config file"),
) -> None:
    """List staged (pending approval) skills."""
    cfg = _load_config(config)
    staging_dir = Path(cfg.skills.builder.staging_dir)

    if not staging_dir.is_dir():
        console.print("[dim]No staged skills.[/dim]")
        return

    found = False
    for entry in sorted(staging_dir.iterdir()):
        if entry.is_dir() and not entry.name.startswith("."):
            found = True
            marker = entry / ".auto_generated"
            desc = ""
            if marker.is_file():
                import json

                try:
                    data = json.loads(marker.read_text(encoding="utf-8"))
                    desc = data.get("description", "")
                except Exception:
                    pass
            console.print(f"  [bold]{entry.name}[/bold]  {desc}")

    if not found:
        console.print("[dim]No staged skills.[/dim]")


@skills_app.command("approve")
def skills_approve(
    name: str = typer.Argument(help="Name of staged skill to approve"),
    config: str | None = typer.Option(None, "--config", "-c", help="Path to config file"),
) -> None:
    """Approve a staged skill and activate it."""
    import shutil

    cfg = _load_config(config)
    staging_path = Path(cfg.skills.builder.staging_dir) / name
    if not staging_path.is_dir():
        console.print(f"[red]No staged skill: {name}[/red]")
        raise typer.Exit(1)

    target = Path(cfg.skills.directory) / name
    if target.exists():
        console.print(f"[red]Skill '{name}' already exists at {target}[/red]")
        raise typer.Exit(1)

    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(staging_path), str(target))
    console.print(f"[green]Skill '{name}' approved and moved to {target}[/green]")


@skills_app.command("reject")
def skills_reject(
    name: str = typer.Argument(help="Name of staged skill to reject"),
    config: str | None = typer.Option(None, "--config", "-c", help="Path to config file"),
) -> None:
    """Reject and delete a staged skill."""
    import shutil

    cfg = _load_config(config)
    staging_path = Path(cfg.skills.builder.staging_dir) / name
    if not staging_path.is_dir():
        console.print(f"[red]No staged skill: {name}[/red]")
        raise typer.Exit(1)

    shutil.rmtree(staging_path)
    console.print(f"[yellow]Staged skill '{name}' deleted.[/yellow]")
