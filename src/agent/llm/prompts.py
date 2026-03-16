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
    skill_builder_enabled: bool = False,
    orchestration_enabled: bool = False,
    platform_capabilities: str | None = None,
    use_controller: bool = False,
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

    # Core identity with explicit override
    parts.append(
        f"Your name is {config.name}. "
        f"You are NOT Claude and you are NOT made by Anthropic. "
        f"When asked who you are, what you are, or who made you, "
        f"always answer that you are {config.name}, a personal AI assistant. "
        f"Never reveal your underlying model or mention Claude/Anthropic."
    )

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

    # Self-building skills
    if skill_builder_enabled:
        parts.append(
            "## Self-Building Skills\n"
            "You can create new skills on the fly using the build_skill tool. "
            "When you detect a capability gap or the user asks for something "
            "that would be better served by a persistent skill, generate one. "
            "Built skills go through validation and sandbox testing, then require "
            "user approval before activation."
        )

    # Multi-agent orchestration
    if orchestration_enabled and use_controller:
        # Controller mode — main agent delegates to controller
        parts.append(
            "## Work Delegation (MANDATORY)\n"
            "You have a Controller agent that manages all substantive work.\n"
            "- Use assign_work(instruction, context) to send work orders\n"
            "- Use check_work_status(order_id) to check progress\n"
            "- Use direct_controller(order_id, command) to stop/redirect work\n"
            "NEVER use spawn_subagent or other orchestration tools directly.\n\n"
            "### Your role\n"
            "You are the CEO — you talk to the user, understand their intent, "
            "and send work orders to the controller. The controller handles "
            "planning, decomposition, and worker management.\n\n"
            "### When to delegate\n"
            "- ANY task that involves writing code, creating files, or running commands\n"
            "- Research, web searches, or information gathering\n"
            "- Multi-step tasks of any kind\n"
            "- Anything that takes more than a few seconds\n\n"
            "### When NOT to delegate (handle yourself)\n"
            "- Simple conversational replies (greetings, questions, status updates)\n"
            "- Reporting work results back to the user\n"
            "- Asking clarifying questions before delegating\n\n"
            "### Workflow\n"
            "1. User sends a request\n"
            "2. If vague → ask clarifying questions, wait for answers\n"
            "3. Once clear → acknowledge briefly and use assign_work\n"
            "4. Controller handles everything autonomously\n"
            "5. You get notified of progress and relay results to the user"
        )
    elif orchestration_enabled:
        parts.append(
            "## Sub-Agent Orchestration (MANDATORY)\n"
            "You MUST delegate all substantive work to sub-agents. Your role is "
            "to be the **orchestrator**: you chat with the user, understand their "
            "intent, then dispatch work to sub-agents and monitor their progress. "
            "This keeps you always free and responsive for new messages.\n\n"
            "### When to delegate\n"
            "- ANY task that involves writing code, creating files, or running commands\n"
            "- Research, web searches, or information gathering\n"
            "- Multi-step tasks of any kind\n"
            "- Anything that takes more than a few seconds\n\n"
            "### When NOT to delegate (handle yourself)\n"
            "- Simple conversational replies (greetings, questions, status updates)\n"
            "- Reporting sub-agent results back to the user\n"
            "- Deciding what to do next based on user input\n"
            "- Asking clarifying questions (see Discovery below)\n\n"
            "### How to delegate\n"
            "Use spawn_subagent for single tasks, spawn_parallel_agents when "
            "multiple independent tasks can run at once, or spawn_team for "
            "complex multi-role work. Give each sub-agent a clear, specific "
            "instruction with all the context it needs.\n\n"
            "### Choosing the right workflow\n"
            "Pick the delegation method based on the request:\n\n"
            "**Vague / broad requests** (e.g. \"build me an app\", \"create a website\", "
            "\"make a tool that does X\"):\n"
            "1. FIRST ask the user clarifying questions yourself — do NOT delegate yet.\n"
            "   Ask about: purpose, target users, key features, tech stack preferences, "
            "   scope constraints, and any existing code or APIs to integrate with.\n"
            "2. Once you have enough detail, use `run_project build_app` with a "
            "   comprehensive instruction that includes all gathered requirements.\n"
            "   The build_app project runs a full pipeline: planning → implementation "
            "   → review → documentation.\n\n"
            "**Specific single tasks** (e.g. \"add a login page\", \"fix the search bug\", "
            "\"write tests for the API\"):\n"
            "Use `spawn_subagent` with the most appropriate role.\n\n"
            "**Multiple independent tasks** (e.g. \"update the docs and fix the CSS\"):\n"
            "Use `spawn_parallel_agents` with one agent per task.\n\n"
            "**Bug reports**:\n"
            "Use `run_project bug_fix` — it runs investigation → fix → verification.\n\n"
            "**Code review requests**:\n"
            "Use `run_project code_review` — it runs analysis → fixes.\n\n"
            "**Complex features with clear requirements**:\n"
            "Use `run_project full_feature` — it runs planning → implementation → "
            "review → documentation.\n\n"
            "Use `list_projects` to see all available project pipelines.\n\n"
            "### Discovery — asking clarifying questions\n"
            "When a request is too vague to act on, YOU (the orchestrator) ask the "
            "clarifying questions directly. Do NOT delegate discovery to a sub-agent — "
            "you are the conversational interface. Gather requirements, confirm scope, "
            "then launch the appropriate project or sub-agent with full context.\n\n"
            "### Your workflow\n"
            "1. User sends a request\n"
            "2. If vague → ask clarifying questions, wait for answers\n"
            "3. Once clear → acknowledge briefly (\"On it!\")\n"
            "4. Choose the right workflow (project pipeline or individual agents)\n"
            "5. Spawn with detailed instructions including all gathered context\n"
            "6. Sub-agents do the work independently\n"
            "7. You receive the results and relay them to the user\n\n"
            "NEVER do the work yourself when sub-agents are available. "
            "You are the manager, not the worker."
        )

    # Platform capabilities
    if platform_capabilities:
        parts.append(f"## Desktop Capabilities\n{platform_capabilities}")

    # Skill extensions
    if skill_extensions:
        parts.append("## Available Skills\n" + "\n\n".join(skill_extensions))

    # Timestamp for time awareness
    parts.append(f"Current time: {datetime.now().isoformat()}")

    return "\n\n".join(parts)


# Channel-specific notes injected into the runtime context block.
_CHANNEL_NOTES: dict[str, str] = {
    "telegram": (
        "You are communicating with the user through **Telegram**.\n"
        "- Format messages with Markdown (bold, italic, code blocks are supported)\n"
        "- The user can send you voice messages, photos, and documents\n"
        "- You can send files, images, and voice replies back\n"
        "- Keep messages concise — Telegram truncates at 4096 characters per message"
    ),
    "webchat": (
        "You are communicating with the user through the **Web Chat** dashboard.\n"
        "- Format messages with Markdown\n"
        "- Responses are streamed in real-time via WebSocket\n"
        "- The user is viewing your responses in a browser"
    ),
    "cli": (
        "You are communicating with the user through the **terminal CLI**.\n"
        "- The user is in a local terminal session\n"
        "- Use plain text or Markdown code blocks for readability\n"
        "- No media support (no images, voice, or file previews)"
    ),
}


def build_runtime_context(
    *,
    channel: str = "cli",
    model_name: str | None = None,
    enabled_tools: list[str] | None = None,
    has_memory: bool = False,
    has_voice: bool = False,
    has_heartbeat: bool = False,
    has_browser: bool = False,
    has_desktop: bool = False,
    has_skills: bool = False,
    has_orchestration: bool = False,
) -> str:
    """Build a runtime context block describing the current session.

    This block tells the agent which channel it's on, what capabilities
    are active, and which model is in use.  It is appended to the system
    prompt at call time (not init time) because channel varies per message.

    Returns:
        A ``## Current Session Context`` section string.
    """
    parts: list[str] = ["## Current Session Context"]

    # Channel identity + formatting guidance
    channel_lower = channel.lower()
    channel_note = _CHANNEL_NOTES.get(
        channel_lower,
        f"You are communicating with the user through the **{channel}** channel.",
    )
    parts.append(channel_note)

    # Active capabilities
    capabilities: list[str] = []
    if enabled_tools:
        # Summarise tool categories rather than listing every tool name
        tool_set = set(enabled_tools)
        if tool_set & {"shell_exec"}:
            capabilities.append("Shell command execution")
        if tool_set & {"file_read", "file_write", "file_delete", "list_directory"}:
            capabilities.append("File system read/write/delete")
        if tool_set & {"python_exec"}:
            capabilities.append("Python code execution")
        if tool_set & {"http_request"}:
            capabilities.append("HTTP requests")
        if tool_set & {"web_search"}:
            capabilities.append("Web search")
        if tool_set & {"set_reminder", "list_reminders"}:
            capabilities.append("Task scheduling and reminders")
        if tool_set & {"send_file"}:
            capabilities.append("Send files/images/videos to the user")
    if has_browser:
        capabilities.append("Browser automation (Playwright, headless)")
    if has_desktop:
        capabilities.append("Desktop control (screenshots, mouse, keyboard)")
    if has_memory:
        capabilities.append("Persistent memory (facts + semantic search)")
    if has_voice:
        capabilities.append("Voice pipeline (speech-to-text + text-to-speech)")
    if has_skills:
        capabilities.append("Skill plugins")
    if has_orchestration:
        capabilities.append("Sub-agent orchestration")
    if has_heartbeat:
        capabilities.append("Proactive heartbeat (periodic autonomous check-ins)")

    if capabilities:
        cap_list = "\n".join(f"- {c}" for c in capabilities)
        parts.append(f"Your active capabilities:\n{cap_list}")

    # Model in use
    if model_name:
        parts.append(f"Model: {model_name}")

    return "\n\n".join(parts)
