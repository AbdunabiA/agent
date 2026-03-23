"""Claude SDK service package."""

# Mutable module globals (_SDK_AVAILABLE, _SDK_PATCHED) live in _core.py
# and are reassigned at runtime.  A plain ``from _core import _SDK_PATCHED``
# would snapshot the initial ``False`` value and never see updates.
# We use ``__getattr__`` so that ``claude_sdk._SDK_PATCHED`` always reads
# the *current* value from the canonical source.
from agent.llm.claude_sdk import _core as _core_mod
from agent.llm.claude_sdk._core import (
    ClaudeSDKService,
    PermissionCallback,
    QuestionCallback,
    SDKStreamEvent,
    SDKTaskResult,
    SDKTaskStatus,
    _format_tool_details,
    _patch_sdk_permission_protocol,
    sdk_available,
)


def __getattr__(name: str):
    if name in ("_SDK_AVAILABLE", "_SDK_PATCHED"):
        return getattr(_core_mod, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "ClaudeSDKService",
    "PermissionCallback",
    "QuestionCallback",
    "SDKStreamEvent",
    "SDKTaskResult",
    "SDKTaskStatus",
    "_SDK_AVAILABLE",
    "_SDK_PATCHED",
    "_format_tool_details",
    "_patch_sdk_permission_protocol",
    "sdk_available",
]
