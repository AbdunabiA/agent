"""Voice pipeline and daemon service CLI commands."""

from __future__ import annotations

import asyncio
from pathlib import Path

import typer
from rich.panel import Panel
from rich.table import Table

from agent.cli._helpers import _load_config, console
from agent.core.events import EventBus

voice_app = typer.Typer(help="Voice pipeline commands.")
daemon_app = typer.Typer(help="Daemon service management (launchd/systemd).")


# --- Voice commands ---


@voice_app.command("list-voices")
def voice_list_voices(
    language: str = typer.Option(
        "", "--language", "-l", help="Filter by language code (en, ru, uz)"
    ),
    config: str | None = typer.Option(None, "--config", "-c", help="Path to config file"),
) -> None:
    """List available TTS voices."""
    cfg = _load_config(config)

    async def _run() -> None:
        from agent.voice.pipeline import VoicePipeline

        pipeline = VoicePipeline(cfg.voice, EventBus())
        voices = await pipeline.list_voices(language)

        if not voices:
            console.print("[yellow]No voices found.[/yellow]")
            return

        table = Table(title=f"TTS Voices ({cfg.voice.tts.provider})")
        table.add_column("Name", style="green")
        table.add_column("Gender")
        table.add_column("Language")

        for v in voices:
            table.add_row(
                v.get("name", ""),
                v.get("gender", ""),
                v.get("language", ""),
            )

        console.print(table)
        console.print(f"\n[dim]Total: {len(voices)} voices[/dim]")

    asyncio.run(_run())


@voice_app.command("test")
def voice_test(
    text: str = typer.Argument(..., help="Text to synthesize"),
    config: str | None = typer.Option(None, "--config", "-c", help="Path to config file"),
) -> None:
    """Synthesize text to speech and save the audio file."""
    cfg = _load_config(config)

    async def _run() -> None:
        from agent.voice.pipeline import VoicePipeline

        pipeline = VoicePipeline(cfg.voice, EventBus())
        result = await pipeline.synthesize(text)

        if not result:
            console.print("[red]TTS synthesis failed or TTS is disabled.[/red]")
            return

        ext = "ogg" if result.mime_type == "audio/ogg" else "mp3"
        out_path = Path(f"data/voice_test.{ext}")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        await asyncio.to_thread(out_path.write_bytes, result.audio_data)

        console.print(
            Panel(
                f"Voice: [green]{result.voice}[/green]\n"
                f"Size: {len(result.audio_data):,} bytes\n"
                f"Duration: ~{result.duration_seconds:.1f}s\n"
                f"Saved to: [blue]{out_path}[/blue]",
                title="TTS Result",
            )
        )

    asyncio.run(_run())


@voice_app.command("config")
def voice_config_cmd(
    config: str | None = typer.Option(None, "--config", "-c", help="Path to config file"),
) -> None:
    """Show current voice configuration."""
    cfg = _load_config(config)
    vc = cfg.voice

    table = Table(title="Voice Configuration")
    table.add_column("Setting", style="cyan")
    table.add_column("Value")

    table.add_row("STT Provider", vc.stt.provider)
    table.add_row("STT Language", vc.stt.language or "(auto-detect)")
    table.add_row("TTS Enabled", str(vc.tts.enabled))
    table.add_row("TTS Provider", vc.tts.provider)
    table.add_row("TTS Voice", vc.tts.edge_voice)
    table.add_row("TTS Rate", vc.tts.edge_rate)
    table.add_row("Output Format", vc.tts.output_format)
    table.add_row("Auto Voice Reply", str(vc.auto_voice_reply))
    table.add_row("Voice Reply Channels", ", ".join(vc.voice_reply_channels))

    console.print(table)


# --- Daemon commands ---


@daemon_app.command("install")
def daemon_install_cmd() -> None:
    """Install the agent as an OS service (launchd on macOS, systemd on Linux)."""
    from agent.core.daemon import daemon_install

    result = daemon_install()
    console.print(result)


@daemon_app.command("uninstall")
def daemon_uninstall_cmd() -> None:
    """Uninstall the agent OS service."""
    from agent.core.daemon import daemon_uninstall

    result = daemon_uninstall()
    console.print(result)


@daemon_app.command("start")
def daemon_start_cmd() -> None:
    """Start the agent daemon service."""
    from agent.core.daemon import daemon_start

    result = daemon_start()
    console.print(result)


@daemon_app.command("stop")
def daemon_stop_cmd() -> None:
    """Stop the agent daemon service."""
    from agent.core.daemon import daemon_stop

    result = daemon_stop()
    console.print(result)


@daemon_app.command("restart")
def daemon_restart_cmd() -> None:
    """Restart the agent daemon service."""
    from agent.core.daemon import daemon_restart

    result = daemon_restart()
    console.print(result)


@daemon_app.command("status")
def daemon_status_cmd() -> None:
    """Show the current daemon service status."""
    from agent.core.daemon import daemon_status

    status = daemon_status()

    table = Table(title="Daemon Status")
    table.add_column("Property", style="cyan")
    table.add_column("Value")

    table.add_row("Installed", "[green]Yes[/green]" if status.installed else "[red]No[/red]")
    table.add_row("Running", "[green]Yes[/green]" if status.running else "[red]No[/red]")
    if status.pid:
        table.add_row("PID", str(status.pid))
    if status.service_path:
        table.add_row("Service File", status.service_path)
    if status.log_path:
        table.add_row("Log Directory", status.log_path)

    console.print(table)
