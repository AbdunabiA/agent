"""Desktop control — screen capture, mouse, keyboard, apps, windows, vision.

Provides cross-platform GUI automation using pyautogui, Pillow,
and platform-specific window management APIs.
"""

from agent.desktop.platform_utils import OSType, PlatformInfo, get_platform

__all__ = [
    "OSType",
    "PlatformInfo",
    "get_platform",
]
