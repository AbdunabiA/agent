"""Cross-platform keyboard control.

Uses pyautogui for key simulation. Handles platform-specific modifier mapping
(Ctrl on Linux/Windows <-> Cmd on macOS).
"""

from __future__ import annotations

import asyncio

import structlog

from agent.desktop.errors import desktop_op
from agent.desktop.platform_utils import OSType, get_hotkey_modifier, get_platform

logger = structlog.get_logger(__name__)


def _build_shortcut_map() -> dict[str, list[str]]:
    """Build the smart shortcut map using current platform's modifier."""
    mod = get_hotkey_modifier()
    info = get_platform()

    close_win = [mod, "q"] if info.os_type == OSType.MACOS else ["alt", "F4"]
    screenshot = [mod, "shift", "3"] if info.os_type == OSType.MACOS else ["printscreen"]

    return {
        "copy": [mod, "c"],
        "paste": [mod, "v"],
        "cut": [mod, "x"],
        "undo": [mod, "z"],
        "redo": [mod, "shift", "z"],
        "save": [mod, "s"],
        "select_all": [mod, "a"],
        "find": [mod, "f"],
        "new_tab": [mod, "t"],
        "close_tab": [mod, "w"],
        "switch_tab": [mod, "tab"],
        "new_window": [mod, "n"],
        "close_window": close_win,
        "refresh": [mod, "r"],
        "address_bar": [mod, "l"],
        "screenshot": screenshot,
    }


def _require_keyboard() -> None:
    """Raise ImportError if no keyboard tool is available."""
    info = get_platform()
    if not info.has_pyautogui and not info.has_wtype:
        raise ImportError(
            "No keyboard tool available. "
            "Install pyautogui (pip install 'agent-ai[desktop]') "
            "or wtype (Wayland)."
        )


@desktop_op("keyboard_type")
async def type_text(text: str, interval: float = 0.02) -> str:
    """Type text as if from keyboard.

    ASCII text uses pyautogui.typewrite.
    Unicode text is copied to clipboard and pasted.

    Args:
        text: Text to type.
        interval: Delay between keystrokes in seconds.

    Returns:
        Description of what was typed.
    """
    _require_keyboard()
    info = get_platform()

    # Use wtype on Wayland if available
    if info.display_server == "wayland" and info.has_wtype and not info.has_pyautogui:
        proc = await asyncio.create_subprocess_exec(
            "wtype", text,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        await proc.communicate()
    else:
        loop = asyncio.get_event_loop()

        # Check if text is ASCII-safe
        try:
            text.encode("ascii")
            is_ascii = True
        except UnicodeEncodeError:
            is_ascii = False

        if is_ascii:

            def _type_ascii() -> None:
                import pyautogui

                pyautogui.typewrite(text, interval=interval)

            await loop.run_in_executor(None, _type_ascii)
        else:
            # Unicode: copy to clipboard and paste
            def _type_unicode() -> None:
                import pyautogui
                import pyperclip

                pyperclip.copy(text)
                mod = get_hotkey_modifier()
                pyautogui.hotkey(mod, "v")

            await loop.run_in_executor(None, _type_unicode)

    logger.info("keyboard_typed", length=len(text))
    return f"Typed {len(text)} characters"


@desktop_op("keyboard_press")
async def press_key(key: str) -> str:
    """Press a single key.

    Args:
        key: Key name -- "enter", "tab", "escape", "backspace", "delete",
             "up", "down", "left", "right", "space", "home", "end",
             "pageup", "pagedown", "f1"-"f12", or a single character.

    Returns:
        Description of the key press.
    """
    _require_keyboard()
    info = get_platform()

    if info.display_server == "wayland" and info.has_wtype and not info.has_pyautogui:
        proc = await asyncio.create_subprocess_exec(
            "wtype", "-k", key,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        await proc.communicate()
    else:
        loop = asyncio.get_event_loop()

        def _press() -> None:
            import pyautogui

            pyautogui.press(key)

        await loop.run_in_executor(None, _press)

    logger.info("key_pressed", key=key)
    return f"Pressed {key}"


@desktop_op("keyboard_hotkey")
async def hotkey(*keys: str) -> str:
    """Press a key combination.

    Supports smart shortcuts that auto-resolve per platform:
        hotkey("copy")  -> Ctrl+C on Linux/Windows, Cmd+C on macOS
        hotkey("paste") -> Ctrl+V / Cmd+V
        hotkey("save")  -> Ctrl+S / Cmd+S

    Or explicit keys:
        hotkey("ctrl", "c")
        hotkey("alt", "tab")
        hotkey("ctrl", "shift", "t")

    Args:
        keys: Key names to press together, or a single smart shortcut name.

    Returns:
        Description of the key combination.
    """
    _require_keyboard()
    info = get_platform()

    resolved_keys = keys

    # Check if it's a smart shortcut
    if len(keys) == 1:
        shortcut_map = _build_shortcut_map()
        shortcut_name = keys[0].lower()
        if shortcut_name in shortcut_map:
            resolved_keys = tuple(shortcut_map[shortcut_name])

    if info.display_server == "wayland" and info.has_wtype and not info.has_pyautogui:
        # wtype -M holds modifier, -m releases; -k presses a key
        for key in resolved_keys[:-1]:
            proc = await asyncio.create_subprocess_exec(
                "wtype", "-M", key,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            await proc.communicate()
        proc = await asyncio.create_subprocess_exec(
            "wtype", "-k", resolved_keys[-1],
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        await proc.communicate()
        for key in reversed(resolved_keys[:-1]):
            proc = await asyncio.create_subprocess_exec(
                "wtype", "-m", key,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            await proc.communicate()
    else:
        loop = asyncio.get_event_loop()

        def _hotkey() -> None:
            import pyautogui

            pyautogui.hotkey(*resolved_keys)

        await loop.run_in_executor(None, _hotkey)

    combo = "+".join(resolved_keys)
    logger.info("hotkey_pressed", combo=combo)
    return f"Pressed {combo}"


@desktop_op("keyboard_hold")
async def hold_key(key: str, duration: float = 0.5) -> str:
    """Hold a key for a duration.

    Args:
        key: Key to hold.
        duration: How long to hold in seconds.

    Returns:
        Description of the action.
    """
    _require_keyboard()
    info = get_platform()

    if info.display_server == "wayland" and info.has_wtype and not info.has_pyautogui:
        proc = await asyncio.create_subprocess_exec(
            "wtype", "-M", key,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        await proc.communicate()
        await asyncio.sleep(duration)
        proc = await asyncio.create_subprocess_exec(
            "wtype", "-m", key,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        await proc.communicate()
    else:
        import time

        loop = asyncio.get_event_loop()

        def _hold() -> None:
            import pyautogui

            pyautogui.keyDown(key)
            time.sleep(duration)
            pyautogui.keyUp(key)

        await loop.run_in_executor(None, _hold)

    logger.info("key_held", key=key, duration=duration)
    return f"Held {key} for {duration}s"
