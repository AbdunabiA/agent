"""Teams and project pipeline CLI commands."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import typer
from rich.panel import Panel
from rich.table import Table

from agent.cli._helpers import _load_config, console

teams_app = typer.Typer(help="Sub-agent team commands.")
project_app = typer.Typer(help="Cross-team project pipeline commands.")


def _load_all_teams(cfg: Any) -> list[Any]:
    """Load teams from both config and teams/ directory."""
    from agent.teams.loader import (
        config_to_team,
        load_teams_from_directory,
        merge_teams,
    )

    config_teams = [config_to_team(t) for t in cfg.orchestration.teams]
    file_teams = load_teams_from_directory(cfg.orchestration.teams_directory)
    return merge_teams(file_teams, config_teams)


# --- Teams subcommands ---


@teams_app.command("list")
def teams_list(
    config: str | None = typer.Option(None, "--config", "-c", help="Path to config file"),
) -> None:
    """List configured sub-agent teams."""
    cfg = _load_config(config)
    teams = _load_all_teams(cfg)

    if not teams:
        console.print(
            "[dim]No teams configured. "
            "Add YAML files to teams/ or define inline in agent.yaml.[/dim]"
        )
        return

    table = Table(title="Agent Teams")
    table.add_column("Name", style="cyan")
    table.add_column("Description")
    table.add_column("Roles")
    table.add_column("#", justify="right")

    for team in teams:
        roles = ", ".join(r.name for r in team.roles)
        table.add_row(team.name, team.description, roles, str(len(team.roles)))

    console.print(table)


@teams_app.command("info")
def teams_info(
    name: str = typer.Argument(help="Team name"),
    config: str | None = typer.Option(None, "--config", "-c", help="Path to config file"),
) -> None:
    """Show details about a sub-agent team."""
    cfg = _load_config(config)
    teams = _load_all_teams(cfg)

    team = next((t for t in teams if t.name == name), None)
    if not team:
        console.print(f"[red]Team '{name}' not found[/red]")
        raise typer.Exit(1)

    console.print(Panel(f"[bold]{team.name}[/bold]\n{team.description}", title="Team"))
    for role in team.roles:
        tools = ", ".join(role.allowed_tools) if role.allowed_tools else "all safe+moderate"
        denied = ", ".join(role.denied_tools) if role.denied_tools else "none"
        console.print(
            f"  [cyan]{role.name}[/cyan]: {role.persona}\n"
            f"    Allowed: {tools} | Denied: {denied} | Max iterations: {role.max_iterations}"
        )


@teams_app.command("create")
def teams_create(
    name: str = typer.Argument(help="Team name"),
    config: str | None = typer.Option(None, "--config", "-c", help="Path to config file"),
) -> None:
    """Scaffold a new team YAML file in the teams/ directory."""
    cfg = _load_config(config)
    teams_dir = Path(cfg.orchestration.teams_directory)
    teams_dir.mkdir(parents=True, exist_ok=True)

    target = teams_dir / f"{name}.yaml"
    if target.exists():
        console.print(f"[red]Team file already exists: {target}[/red]")
        raise typer.Exit(1)

    template = (
        f"name: {name}\n"
        f'description: ""\n'
        f"roles:\n"
        f"  - name: agent\n"
        f'    persona: "You are a helpful assistant."\n'
        f"    allowed_tools: []\n"
        f"    denied_tools: []\n"
        f"    max_iterations: 5\n"
    )
    target.write_text(template, encoding="utf-8")
    console.print(f"[green]Created team file: {target}[/green]")
    console.print("[dim]Edit the file to add roles and configure the team.[/dim]")


# --- Project subcommands ---


@project_app.command("list")
def project_list(
    config: str | None = typer.Option(None, "--config", "-c", help="Path to config file"),
) -> None:
    """List available project pipelines."""
    cfg = _load_config(config)

    from agent.teams.loader import load_projects_from_directory

    projects = load_projects_from_directory(cfg.orchestration.teams_directory)

    if not projects:
        console.print(
            "[dim]No projects configured. "
            "Add YAML files to teams/projects/ to define pipelines.[/dim]"
        )
        return

    table = Table(title="Project Pipelines")
    table.add_column("Name", style="cyan")
    table.add_column("Description")
    table.add_column("Stages")
    table.add_column("Agents", justify="right")

    for proj in projects:
        stages = " → ".join(s.name for s in proj.stages)
        total_agents = sum(len(s.agents) for s in proj.stages)
        table.add_row(proj.name, proj.description, stages, str(total_agents))

    console.print(table)


@project_app.command("info")
def project_info(
    name: str = typer.Argument(help="Project name"),
    config: str | None = typer.Option(None, "--config", "-c", help="Path to config file"),
) -> None:
    """Show details about a project pipeline."""
    cfg = _load_config(config)

    from agent.teams.loader import load_projects_from_directory

    projects = load_projects_from_directory(cfg.orchestration.teams_directory)
    proj = next((p for p in projects if p.name == name), None)

    if not proj:
        console.print(f"[red]Project '{name}' not found[/red]")
        raise typer.Exit(1)

    console.print(Panel(f"[bold]{proj.name}[/bold]\n{proj.description}", title="Project"))
    for i, stage in enumerate(proj.stages, 1):
        agents_str = ", ".join(f"{a.team}/{a.role}" for a in stage.agents)
        parallel_str = "parallel" if stage.parallel else "sequential"
        console.print(
            f"  [cyan]Stage {i}: {stage.name}[/cyan] ({parallel_str})\n" f"    Agents: {agents_str}"
        )


@project_app.command("create")
def project_create(
    name: str = typer.Argument(help="Project name"),
    config: str | None = typer.Option(None, "--config", "-c", help="Path to config file"),
) -> None:
    """Scaffold a new project YAML file in teams/projects/."""
    cfg = _load_config(config)
    projects_dir = Path(cfg.orchestration.teams_directory) / "projects"
    projects_dir.mkdir(parents=True, exist_ok=True)

    target = projects_dir / f"{name}.yaml"
    if target.exists():
        console.print(f"[red]Project file already exists: {target}[/red]")
        raise typer.Exit(1)

    template = (
        f"name: {name}\n"
        f'description: ""\n'
        f"stages:\n"
        f"  - name: planning\n"
        f"    parallel: true\n"
        f"    agents:\n"
        f"      - team: engineering\n"
        f"        role: architect\n"
        f"\n"
        f"  - name: implementation\n"
        f"    parallel: true\n"
        f"    agents:\n"
        f"      - team: engineering\n"
        f"        role: backend_developer\n"
        f"\n"
        f"  - name: review\n"
        f"    parallel: true\n"
        f"    agents:\n"
        f"      - team: quality\n"
        f"        role: qa_engineer\n"
    )
    target.write_text(template, encoding="utf-8")
    console.print(f"[green]Created project file: {target}[/green]")
    console.print("[dim]Edit the file to define stages and agents.[/dim]")
