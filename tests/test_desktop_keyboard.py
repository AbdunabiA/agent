"""Tests for desktop keyboard control."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from agent.desktop.keyboard import (
    _build_shortcut_map,
    hold_key,
    hotkey,
    press_key,
    type_text,
)
from agent.desktop.platform_utils import OSType, PlatformInfo

_PLATFORM_WIN = PlatformInfo(
    os_type=OSType.WINDOWS, has_display=True, display_server="win32",
    has_pyautogui=True, has_wmctrl=False, has_xdotool=False,
    has_osascript=False, screen_width=1920, screen_height=1080, scale_factor=1.0,
)

_PLATFORM_MAC = PlatformInfo(
    os_type=OSType.MACOS, has_display=True, display_server="quartz",
    has_pyautogui=True, has_wmctrl=False, has_xdotool=False,
    has_osascript=True, screen_width=1920, screen_height=1080, scale_factor=1.0,
)


class TestBuildShortcutMap:
    """Tests for smart shortcut resolution."""

    @patch("agent.desktop.keyboard.get_platform", return_value=_PLATFORM_WIN)
    @patch("agent.desktop.keyboard.get_hotkey_modifier", return_value="ctrl")
    def test_copy_resolves_to_ctrl_c_on_windows(self, _1: object, _2: object) -> None:
        shortcut_map = _build_shortcut_map()
        assert shortcut_map["copy"] == ["ctrl", "c"]
        assert shortcut_map["paste"] == ["ctrl", "v"]
        assert shortcut_map["save"] == ["ctrl", "s"]

    @patch("agent.desktop.keyboard.get_platform", return_value=_PLATFORM_MAC)
    @patch("agent.desktop.keyboard.get_hotkey_modifier", return_value="command")
    def test_copy_resolves_to_cmd_c_on_macos(self, _1: object, _2: object) -> None:
        shortcut_map = _build_shortcut_map()
        assert shortcut_map["copy"] == ["command", "c"]
        assert shortcut_map["paste"] == ["command", "v"]
        assert shortcut_map["save"] == ["command", "s"]

    @patch("agent.desktop.keyboard.get_platform", return_value=_PLATFORM_MAC)
    @patch("agent.desktop.keyboard.get_hotkey_modifier", return_value="command")
    def test_close_window_uses_cmd_q_on_macos(self, _1: object, _2: object) -> None:
        shortcut_map = _build_shortcut_map()
        assert shortcut_map["close_window"] == ["command", "q"]

    @patch("agent.desktop.keyboard.get_platform", return_value=_PLATFORM_WIN)
    @patch("agent.desktop.keyboard.get_hotkey_modifier", return_value="ctrl")
    def test_close_window_uses_alt_f4_on_windows(self, _1: object, _2: object) -> None:
        shortcut_map = _build_shortcut_map()
        assert shortcut_map["close_window"] == ["alt", "F4"]


class TestTypeText:
    """Tests for type_text()."""

    @patch("agent.desktop.keyboard.get_platform", return_value=_PLATFORM_WIN)
    async def test_types_ascii_via_typewrite(self, _: object) -> None:
        mock_pyautogui = MagicMock()

        with patch.dict("sys.modules", {"pyautogui": mock_pyautogui}):
            result = await type_text("hello world", interval=0.01)
            assert "11 characters" in result
            mock_pyautogui.typewrite.assert_called_once_with("hello world", interval=0.01)

    @patch("agent.desktop.keyboard.get_platform", return_value=_PLATFORM_WIN)
    @patch("agent.desktop.keyboard.get_hotkey_modifier", return_value="ctrl")
    async def test_types_unicode_via_clipboard(self, _1: object, _2: object) -> None:
        mock_pyautogui = MagicMock()
        mock_pyperclip = MagicMock()

        with patch.dict("sys.modules", {
            "pyautogui": mock_pyautogui,
            "pyperclip": mock_pyperclip,
        }):
            # Use actual non-ASCII text to trigger clipboard path
            unicode_text = "\u041f\u0440\u0438\u0432\u0435\u0442 \u043c\u0438\u0440"
            result = await type_text(unicode_text)
            assert f"{len(unicode_text)} characters" in result
            mock_pyperclip.copy.assert_called_once()
            mock_pyautogui.hotkey.assert_called_once_with("ctrl", "v")


class TestPressKey:
    """Tests for press_key()."""

    @patch("agent.desktop.keyboard.get_platform", return_value=_PLATFORM_WIN)
    async def test_presses_key(self, _: object) -> None:
        mock_pyautogui = MagicMock()

        with patch.dict("sys.modules", {"pyautogui": mock_pyautogui}):
            result = await press_key("enter")
            assert "Pressed enter" in result
            mock_pyautogui.press.assert_called_once_with("enter")

    @patch("agent.desktop.keyboard.get_platform", return_value=_PLATFORM_WIN)
    async def test_presses_function_key(self, _: object) -> None:
        mock_pyautogui = MagicMock()

        with patch.dict("sys.modules", {"pyautogui": mock_pyautogui}):
            result = await press_key("f5")
            assert "Pressed f5" in result


class TestHotkey:
    """Tests for hotkey()."""

    @patch("agent.desktop.keyboard.get_platform", return_value=_PLATFORM_WIN)
    @patch("agent.desktop.keyboard.get_hotkey_modifier", return_value="ctrl")
    async def test_explicit_key_combo(self, _1: object, _2: object) -> None:
        mock_pyautogui = MagicMock()

        with patch.dict("sys.modules", {"pyautogui": mock_pyautogui}):
            result = await hotkey("ctrl", "c")
            assert "ctrl+c" in result
            mock_pyautogui.hotkey.assert_called_once_with("ctrl", "c")

    @patch("agent.desktop.keyboard.get_platform", return_value=_PLATFORM_WIN)
    @patch("agent.desktop.keyboard.get_hotkey_modifier", return_value="ctrl")
    async def test_smart_shortcut_copy(self, _1: object, _2: object) -> None:
        mock_pyautogui = MagicMock()

        with patch.dict("sys.modules", {"pyautogui": mock_pyautogui}):
            result = await hotkey("copy")
            assert "ctrl+c" in result

    @patch("agent.desktop.keyboard.get_platform", return_value=_PLATFORM_MAC)
    @patch("agent.desktop.keyboard.get_hotkey_modifier", return_value="command")
    async def test_smart_shortcut_paste_macos(self, _1: object, _2: object) -> None:
        mock_pyautogui = MagicMock()

        with patch.dict("sys.modules", {"pyautogui": mock_pyautogui}):
            result = await hotkey("paste")
            assert "command+v" in result

    @patch("agent.desktop.keyboard.get_platform", return_value=_PLATFORM_WIN)
    @patch("agent.desktop.keyboard.get_hotkey_modifier", return_value="ctrl")
    async def test_alt_tab(self, _1: object, _2: object) -> None:
        mock_pyautogui = MagicMock()

        with patch.dict("sys.modules", {"pyautogui": mock_pyautogui}):
            result = await hotkey("alt", "tab")
            assert "alt+tab" in result


class TestHoldKey:
    """Tests for hold_key()."""

    @patch("agent.desktop.keyboard.get_platform", return_value=_PLATFORM_WIN)
    async def test_holds_key(self, _: object) -> None:
        mock_pyautogui = MagicMock()

        with patch.dict("sys.modules", {"pyautogui": mock_pyautogui}):
            result = await hold_key("shift", duration=0.01)
            assert "Held shift" in result
            mock_pyautogui.keyDown.assert_called_once_with("shift")
            mock_pyautogui.keyUp.assert_called_once_with("shift")
