"""Register all built-in tools.

Import this module to register all built-in tools with the global registry.
Tools with missing optional dependencies are skipped gracefully.
"""

import structlog

logger = structlog.get_logger(__name__)

# Always-available tools (only core deps)
from agent.tools.builtins import (  # noqa: F401, E402
    filesystem,
    http,
    memory,
    monitor,
    orchestration,
    python_exec,
    scheduler,
    send_file,
    shell,
    skill_builder,
    system,
    web_search,
)

# Tools with optional dependencies — skip if not installed
_optional_modules = ["browser", "desktop"]

for _mod_name in _optional_modules:
    try:
        __import__(f"agent.tools.builtins.{_mod_name}", fromlist=[_mod_name])
    except ImportError as e:
        logger.debug("builtin_tool_skipped", module=_mod_name, reason=str(e))
