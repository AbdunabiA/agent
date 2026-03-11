"""Common utility functions."""

from __future__ import annotations

import platform
import sys


def get_system_info() -> dict[str, str]:
    """Get basic system information.

    Returns:
        Dictionary with python_version, os, platform keys.
    """
    return {
        "python_version": (
            f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
        ),
        "os": platform.system(),
        "platform": platform.platform(),
        "architecture": platform.machine(),
    }
