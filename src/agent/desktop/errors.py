"""Consistent desktop error handling.

Provides custom exception classes and a decorator that wraps desktop
operations with capability checks and consistent error formatting.
"""

from __future__ import annotations

import functools
from collections.abc import Callable
from typing import Any, TypeVar

import structlog

logger = structlog.get_logger(__name__)

F = TypeVar("F", bound=Callable[..., Any])


class DesktopError(Exception):
    """Base error for desktop operations."""


class PlatformNotSupportedError(DesktopError):
    """Operation not supported on this platform."""


class MissingDependencyError(DesktopError):
    """Required dependency is not installed."""


class NoDisplayError(DesktopError):
    """No display/GUI environment available."""


def desktop_op(feature_name: str) -> Callable[[F], F]:
    """Decorator for desktop operations.

    Checks platform capabilities, catches exceptions, and returns
    consistent ``[UNAVAILABLE] feature: reason`` messages on failure.

    Args:
        feature_name: Human-readable name of the desktop feature.

    Returns:
        Decorator function.
    """

    def decorator(func: F) -> F:
        @functools.wraps(func)
        async def wrapper(*args: Any, **kwargs: Any) -> Any:
            from agent.desktop.platform_utils import get_platform

            info = get_platform()

            # Check display availability
            if not info.has_display:
                return (
                    f"[UNAVAILABLE] {feature_name}: "
                    "No display available. Desktop tools require a GUI environment."
                )

            try:
                return await func(*args, **kwargs)
            except ImportError as e:
                msg = f"[UNAVAILABLE] {feature_name}: Missing dependency — {e}"
                logger.warning("desktop_op_missing_dep", feature=feature_name, error=str(e))
                return msg
            except (PlatformNotSupportedError, MissingDependencyError, NoDisplayError) as e:
                msg = f"[UNAVAILABLE] {feature_name}: {e}"
                logger.warning("desktop_op_unsupported", feature=feature_name, error=str(e))
                return msg
            except Exception as e:
                msg = f"[ERROR] {feature_name}: {e}"
                logger.error("desktop_op_failed", feature=feature_name, error=str(e))
                return msg

        return wrapper  # type: ignore[return-value]

    return decorator
