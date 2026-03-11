"""System prompt templates for the agent.

Builds the system prompt from config and optional soul.md content.
Phase 4 will add soul.md loading; for now, uses config.agent.persona.
"""

from __future__ import annotations

import platform
from datetime import datetime

from agent.config import AgentPersonaConfig


def build_system_prompt(
    config: AgentPersonaConfig,
    soul_content: str | None = None,
    skill_extensions: list[str] | None = None,
) -> str:
    """Build the system prompt for the agent.

    Args:
        config: Agent persona configuration.
        soul_content: Optional soul.md content.
        skill_extensions: Optional prompt extensions from loaded skills.

    Returns:
        Complete system prompt string.
    """
    parts: list[str] = []

    # Core identity
    parts.append(f"Your name is {config.name}.")

    # Persona (from config or soul.md)
    if soul_content:
        parts.append(soul_content)
    else:
        parts.append(config.persona)

    # System info
    os_name = platform.system()  # "Windows", "Linux", "Darwin"

    # Standard instructions
    parts.append(
        "You are running as a local AI assistant on the user's machine. "
        "You have access to tools that let you execute shell commands, "
        "read and write files, browse the web, and more. "
        "When you need to perform an action, use the available tools. "
        "Be concise, helpful, and proactive."
    )

    # OS-specific guidance for opening programs
    if os_name == "Windows":
        parts.append(
            "## System\n"
            "You are running on Windows. To open programs, URLs, or files that the user "
            "can see, use the shell_exec tool with Windows commands:\n"
            "- Open a URL in browser: `start https://example.com`\n"
            "- Open a program: `start notepad` or `start calc`\n"
            "- Open a file: `start \"\" \"C:\\path\\to\\file.txt\"`\n"
            "- Play music in browser: `start https://youtube.com/watch?v=...`\n"
            "The `start` command opens things visibly for the user. "
            "Only use the browser tool (Playwright) when you need to scrape or interact "
            "with web pages programmatically — it runs headless and the user cannot see it."
        )
    elif os_name == "Darwin":
        parts.append(
            "## System\n"
            "You are running on macOS. To open programs, URLs, or files visibly, "
            "use the shell_exec tool with `open`:\n"
            "- Open a URL: `open https://example.com`\n"
            "- Open an app: `open -a Safari` or `open -a 'Visual Studio Code'`\n"
            "- Open a file: `open /path/to/file.txt`\n"
            "Only use the browser tool (Playwright) for headless web scraping."
        )
    else:
        parts.append(
            "## System\n"
            "You are running on Linux. To open programs, URLs, or files visibly, "
            "use the shell_exec tool with `xdg-open`:\n"
            "- Open a URL: `xdg-open https://example.com`\n"
            "- Open a file: `xdg-open /path/to/file.txt`\n"
            "Only use the browser tool (Playwright) for headless web scraping."
        )

    # Skill extensions
    if skill_extensions:
        parts.append("## Available Skills\n" + "\n\n".join(skill_extensions))

    # Timestamp for time awareness
    parts.append(f"Current time: {datetime.now().isoformat()}")

    return "\n\n".join(parts)
