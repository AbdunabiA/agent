"""Register all built-in tools.

Import this module to register all built-in tools with the global registry.
"""

from agent.tools.builtins import (
    browser,  # noqa: F401
    desktop,  # noqa: F401
    filesystem,  # noqa: F401
    http,  # noqa: F401
    memory,  # noqa: F401
    monitor,  # noqa: F401
    orchestration,  # noqa: F401
    python_exec,  # noqa: F401
    scheduler,  # noqa: F401
    send_file,  # noqa: F401
    shell,  # noqa: F401
    skill_builder,  # noqa: F401
    system,  # noqa: F401
    web_search,  # noqa: F401
)
