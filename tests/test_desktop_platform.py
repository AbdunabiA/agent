"""Tests for desktop platform detection utilities."""

from __future__ import annotations

from unittest.mock import patch

from agent.desktop.platform_utils import (
    OSType,
    PlatformInfo,
    detect_platform,
    get_app_launch_command,
    get_hotkey_modifier,
    get_platform,
    reset_platform_cache,
)


class TestDetectPlatform:
    """Tests for detect_platform()."""

    def test_returns_platform_info(self) -> None:
        info = detect_platform()
        assert isinstance(info, PlatformInfo)
        assert info.os_type in OSType
        assert info.screen_width > 0
        assert info.screen_height > 0

    @patch("agent.desktop.platform_utils.platform.system", return_value="Windows")
    def test_detects_windows(self, _mock: object) -> None:
        info = detect_platform()
        assert info.os_type == OSType.WINDOWS
        assert info.display_server == "win32"
        assert info.has_display is True

    @patch("agent.desktop.platform_utils.platform.system", return_value="Darwin")
    def test_detects_macos(self, _mock: object) -> None:
        info = detect_platform()
        assert info.os_type == OSType.MACOS
        assert info.display_server == "quartz"
        assert info.has_display is True

    @patch("agent.desktop.platform_utils.platform.system", return_value="Linux")
    @patch.dict("os.environ", {"DISPLAY": ":0", "WAYLAND_DISPLAY": ""})
    def test_detects_linux_x11(self, _mock: object) -> None:
        info = detect_platform()
        assert info.os_type == OSType.LINUX
        assert info.display_server == "x11"
        assert info.has_display is True

    @patch("agent.desktop.platform_utils.platform.system", return_value="Linux")
    @patch.dict("os.environ", {"WAYLAND_DISPLAY": "wayland-0", "DISPLAY": ""})
    def test_detects_linux_wayland(self, _mock: object) -> None:
        info = detect_platform()
        assert info.os_type == OSType.LINUX
        assert info.display_server == "wayland"
        assert info.has_display is True

    @patch("agent.desktop.platform_utils.platform.system", return_value="Linux")
    @patch.dict("os.environ", {"DISPLAY": "", "WAYLAND_DISPLAY": ""}, clear=False)
    def test_detects_linux_no_display(self, _mock: object) -> None:
        info = detect_platform()
        assert info.os_type == OSType.LINUX
        assert info.has_display is False
        assert info.display_server == ""

    @patch("agent.desktop.platform_utils.platform.system", return_value="FreeBSD")
    def test_detects_unknown_os(self, _mock: object) -> None:
        info = detect_platform()
        assert info.os_type == OSType.UNKNOWN

    def test_platform_info_is_frozen(self) -> None:
        import pytest

        info = detect_platform()
        with pytest.raises(AttributeError):
            info.os_type = OSType.LINUX  # type: ignore[misc]


class TestGetPlatform:
    """Tests for get_platform() caching."""

    def test_returns_same_instance(self) -> None:
        reset_platform_cache()
        a = get_platform()
        b = get_platform()
        assert a is b

    def test_reset_cache_works(self) -> None:
        reset_platform_cache()
        a = get_platform()
        reset_platform_cache()
        b = get_platform()
        # New instances (may be equal but not same object)
        assert a is not b


class TestGetHotkeyModifier:
    """Tests for get_hotkey_modifier()."""

    @patch("agent.desktop.platform_utils._platform_info", PlatformInfo(
        os_type=OSType.MACOS, has_display=True, display_server="quartz",
        has_pyautogui=False, has_wmctrl=False, has_xdotool=False,
        has_osascript=True, screen_width=1920, screen_height=1080, scale_factor=1.0,
    ))
    def test_returns_command_on_macos(self) -> None:
        assert get_hotkey_modifier() == "command"

    @patch("agent.desktop.platform_utils._platform_info", PlatformInfo(
        os_type=OSType.WINDOWS, has_display=True, display_server="win32",
        has_pyautogui=False, has_wmctrl=False, has_xdotool=False,
        has_osascript=False, screen_width=1920, screen_height=1080, scale_factor=1.0,
    ))
    def test_returns_ctrl_on_windows(self) -> None:
        assert get_hotkey_modifier() == "ctrl"

    @patch("agent.desktop.platform_utils._platform_info", PlatformInfo(
        os_type=OSType.LINUX, has_display=True, display_server="x11",
        has_pyautogui=False, has_wmctrl=False, has_xdotool=False,
        has_osascript=False, screen_width=1920, screen_height=1080, scale_factor=1.0,
    ))
    def test_returns_ctrl_on_linux(self) -> None:
        assert get_hotkey_modifier() == "ctrl"


class TestGetAppLaunchCommand:
    """Tests for get_app_launch_command()."""

    @patch("agent.desktop.platform_utils._platform_info", PlatformInfo(
        os_type=OSType.MACOS, has_display=True, display_server="quartz",
        has_pyautogui=False, has_wmctrl=False, has_xdotool=False,
        has_osascript=True, screen_width=1920, screen_height=1080, scale_factor=1.0,
    ))
    def test_macos_uses_open(self) -> None:
        cmd = get_app_launch_command("Safari")
        assert cmd == ["open", "-a", "Safari"]

    @patch("agent.desktop.platform_utils._platform_info", PlatformInfo(
        os_type=OSType.WINDOWS, has_display=True, display_server="win32",
        has_pyautogui=False, has_wmctrl=False, has_xdotool=False,
        has_osascript=False, screen_width=1920, screen_height=1080, scale_factor=1.0,
    ))
    def test_windows_uses_start(self) -> None:
        cmd = get_app_launch_command("notepad")
        assert cmd == ["cmd", "/c", "start", "", "notepad"]

    @patch("agent.desktop.platform_utils._platform_info", PlatformInfo(
        os_type=OSType.LINUX, has_display=True, display_server="x11",
        has_pyautogui=False, has_wmctrl=False, has_xdotool=False,
        has_osascript=False, screen_width=1920, screen_height=1080, scale_factor=1.0,
    ))
    @patch("agent.desktop.platform_utils.shutil.which", return_value=None)
    def test_linux_uses_xdg_open_when_not_in_path(self, _mock: object) -> None:
        cmd = get_app_launch_command("myapp")
        assert cmd == ["xdg-open", "myapp"]

    @patch("agent.desktop.platform_utils._platform_info", PlatformInfo(
        os_type=OSType.LINUX, has_display=True, display_server="x11",
        has_pyautogui=False, has_wmctrl=False, has_xdotool=False,
        has_osascript=False, screen_width=1920, screen_height=1080, scale_factor=1.0,
    ))
    @patch("agent.desktop.platform_utils.shutil.which", return_value="/usr/bin/firefox")
    def test_linux_uses_direct_command_when_in_path(self, _mock: object) -> None:
        cmd = get_app_launch_command("firefox")
        assert cmd == ["firefox"]
