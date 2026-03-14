"""Tests for desktop capabilities detection and summary."""

from __future__ import annotations

from unittest.mock import patch

from agent.desktop.platform_utils import (
    OSType,
    PlatformInfo,
    get_capabilities,
    get_capabilities_summary,
)

# ---------------------------------------------------------------------------
# Helpers — factory functions for common PlatformInfo configurations
# ---------------------------------------------------------------------------

def _linux_x11_with_pyautogui() -> PlatformInfo:
    """Linux X11 with pyautogui, wmctrl, and xdotool available."""
    return PlatformInfo(
        os_type=OSType.LINUX,
        has_display=True,
        display_server="x11",
        has_pyautogui=True,
        has_wmctrl=True,
        has_xdotool=True,
        has_osascript=False,
        screen_width=1920,
        screen_height=1080,
        scale_factor=1.0,
        has_wtype=False,
        has_ydotool=False,
        has_grim=False,
        has_slurp=False,
        has_pygetwindow=False,
        has_uiautomation=False,
        has_pyatspi=False,
        monitor_count=2,
    )


def _linux_wayland_with_native_tools() -> PlatformInfo:
    """Linux Wayland with grim, wtype, ydotool, and slurp."""
    return PlatformInfo(
        os_type=OSType.LINUX,
        has_display=True,
        display_server="wayland",
        has_pyautogui=False,
        has_wmctrl=False,
        has_xdotool=False,
        has_osascript=False,
        screen_width=2560,
        screen_height=1440,
        scale_factor=1.5,
        has_wtype=True,
        has_ydotool=True,
        has_grim=True,
        has_slurp=True,
        has_pygetwindow=False,
        has_uiautomation=False,
        has_pyatspi=False,
        monitor_count=1,
    )


def _linux_wayland_without_native_tools() -> PlatformInfo:
    """Linux Wayland with no native screenshot/input tools."""
    return PlatformInfo(
        os_type=OSType.LINUX,
        has_display=True,
        display_server="wayland",
        has_pyautogui=False,
        has_wmctrl=False,
        has_xdotool=False,
        has_osascript=False,
        screen_width=1920,
        screen_height=1080,
        scale_factor=1.0,
        has_wtype=False,
        has_ydotool=False,
        has_grim=False,
        has_slurp=False,
        has_pygetwindow=False,
        has_uiautomation=False,
        has_pyatspi=False,
        monitor_count=1,
    )


def _linux_wayland_partial_tools() -> PlatformInfo:
    """Linux Wayland with only grim (screenshots) installed."""
    return PlatformInfo(
        os_type=OSType.LINUX,
        has_display=True,
        display_server="wayland",
        has_pyautogui=False,
        has_wmctrl=False,
        has_xdotool=False,
        has_osascript=False,
        screen_width=1920,
        screen_height=1080,
        scale_factor=1.0,
        has_wtype=False,
        has_ydotool=False,
        has_grim=True,
        has_slurp=False,
        has_pygetwindow=False,
        has_uiautomation=False,
        has_pyatspi=False,
        monitor_count=1,
    )


def _windows_with_pygetwindow() -> PlatformInfo:
    """Windows with pyautogui and pygetwindow."""
    return PlatformInfo(
        os_type=OSType.WINDOWS,
        has_display=True,
        display_server="win32",
        has_pyautogui=True,
        has_wmctrl=False,
        has_xdotool=False,
        has_osascript=False,
        screen_width=2560,
        screen_height=1440,
        scale_factor=1.5,
        has_wtype=False,
        has_ydotool=False,
        has_grim=False,
        has_slurp=False,
        has_pygetwindow=True,
        has_uiautomation=True,
        has_pyatspi=False,
        monitor_count=3,
    )


def _windows_without_pygetwindow() -> PlatformInfo:
    """Windows with pyautogui but without pygetwindow."""
    return PlatformInfo(
        os_type=OSType.WINDOWS,
        has_display=True,
        display_server="win32",
        has_pyautogui=True,
        has_wmctrl=False,
        has_xdotool=False,
        has_osascript=False,
        screen_width=1920,
        screen_height=1080,
        scale_factor=1.0,
        has_wtype=False,
        has_ydotool=False,
        has_grim=False,
        has_slurp=False,
        has_pygetwindow=False,
        has_uiautomation=False,
        has_pyatspi=False,
        monitor_count=1,
    )


def _macos() -> PlatformInfo:
    """macOS with osascript and pyautogui."""
    return PlatformInfo(
        os_type=OSType.MACOS,
        has_display=True,
        display_server="quartz",
        has_pyautogui=True,
        has_wmctrl=False,
        has_xdotool=False,
        has_osascript=True,
        screen_width=2560,
        screen_height=1600,
        scale_factor=2.0,
        has_wtype=False,
        has_ydotool=False,
        has_grim=False,
        has_slurp=False,
        has_pygetwindow=False,
        has_uiautomation=False,
        has_pyatspi=False,
        monitor_count=1,
    )


def _no_display() -> PlatformInfo:
    """Headless Linux server with no display."""
    return PlatformInfo(
        os_type=OSType.LINUX,
        has_display=False,
        display_server="",
        has_pyautogui=False,
        has_wmctrl=False,
        has_xdotool=False,
        has_osascript=False,
        screen_width=1920,
        screen_height=1080,
        scale_factor=1.0,
        has_wtype=False,
        has_ydotool=False,
        has_grim=False,
        has_slurp=False,
        has_pygetwindow=False,
        has_uiautomation=False,
        has_pyatspi=False,
        monitor_count=1,
    )


def _unknown_os() -> PlatformInfo:
    """Unknown OS with no capabilities."""
    return PlatformInfo(
        os_type=OSType.UNKNOWN,
        has_display=False,
        display_server="",
        has_pyautogui=False,
        has_wmctrl=False,
        has_xdotool=False,
        has_osascript=False,
        screen_width=1920,
        screen_height=1080,
        scale_factor=1.0,
        has_wtype=False,
        has_ydotool=False,
        has_grim=False,
        has_slurp=False,
        has_pygetwindow=False,
        has_uiautomation=False,
        has_pyatspi=False,
        monitor_count=1,
    )


# ---------------------------------------------------------------------------
# get_capabilities() tests
# ---------------------------------------------------------------------------

class TestGetCapabilitiesLinuxX11:
    """Capabilities on Linux X11 with pyautogui."""

    @patch("agent.desktop.platform_utils._platform_info", _linux_x11_with_pyautogui())
    def test_display_is_true(self) -> None:
        caps = get_capabilities()
        assert caps["display"] is True

    @patch("agent.desktop.platform_utils._platform_info", _linux_x11_with_pyautogui())
    def test_screenshots_via_pyautogui(self) -> None:
        caps = get_capabilities()
        assert caps["screenshots"] is True

    @patch("agent.desktop.platform_utils._platform_info", _linux_x11_with_pyautogui())
    def test_mouse_control_via_pyautogui(self) -> None:
        caps = get_capabilities()
        assert caps["mouse_control"] is True

    @patch("agent.desktop.platform_utils._platform_info", _linux_x11_with_pyautogui())
    def test_keyboard_input_via_pyautogui(self) -> None:
        caps = get_capabilities()
        assert caps["keyboard_input"] is True

    @patch("agent.desktop.platform_utils._platform_info", _linux_x11_with_pyautogui())
    def test_window_management_via_wmctrl_xdotool(self) -> None:
        caps = get_capabilities()
        assert caps["window_management"] is True

    @patch("agent.desktop.platform_utils._platform_info", _linux_x11_with_pyautogui())
    def test_app_launching_always_true(self) -> None:
        caps = get_capabilities()
        assert caps["app_launching"] is True

    @patch("agent.desktop.platform_utils._platform_info", _linux_x11_with_pyautogui())
    def test_wayland_native_false_on_x11(self) -> None:
        caps = get_capabilities()
        assert caps["wayland_native"] is False

    @patch("agent.desktop.platform_utils._platform_info", _linux_x11_with_pyautogui())
    def test_returns_all_expected_keys(self) -> None:
        caps = get_capabilities()
        expected_keys = {
            "display",
            "screenshots",
            "mouse_control",
            "keyboard_input",
            "window_management",
            "app_launching",
            "wayland_native",
            "accessibility_tree",
        }
        assert set(caps.keys()) == expected_keys


class TestGetCapabilitiesLinuxWayland:
    """Capabilities on Linux Wayland with native tools."""

    @patch(
        "agent.desktop.platform_utils._platform_info",
        _linux_wayland_with_native_tools(),
    )
    def test_display_is_true(self) -> None:
        caps = get_capabilities()
        assert caps["display"] is True

    @patch(
        "agent.desktop.platform_utils._platform_info",
        _linux_wayland_with_native_tools(),
    )
    def test_screenshots_via_grim(self) -> None:
        caps = get_capabilities()
        assert caps["screenshots"] is True

    @patch(
        "agent.desktop.platform_utils._platform_info",
        _linux_wayland_with_native_tools(),
    )
    def test_mouse_control_via_ydotool(self) -> None:
        caps = get_capabilities()
        assert caps["mouse_control"] is True

    @patch(
        "agent.desktop.platform_utils._platform_info",
        _linux_wayland_with_native_tools(),
    )
    def test_keyboard_input_via_wtype(self) -> None:
        caps = get_capabilities()
        assert caps["keyboard_input"] is True

    @patch(
        "agent.desktop.platform_utils._platform_info",
        _linux_wayland_with_native_tools(),
    )
    def test_window_management_false_without_tools(self) -> None:
        """No wmctrl/xdotool/osascript/pygetwindow on pure Wayland."""
        caps = get_capabilities()
        assert caps["window_management"] is False

    @patch(
        "agent.desktop.platform_utils._platform_info",
        _linux_wayland_with_native_tools(),
    )
    def test_wayland_native_true_with_grim_and_wtype(self) -> None:
        caps = get_capabilities()
        assert caps["wayland_native"] is True


class TestGetCapabilitiesLinuxWaylandNoTools:
    """Capabilities on Linux Wayland without native tools."""

    @patch(
        "agent.desktop.platform_utils._platform_info",
        _linux_wayland_without_native_tools(),
    )
    def test_display_is_true(self) -> None:
        caps = get_capabilities()
        assert caps["display"] is True

    @patch(
        "agent.desktop.platform_utils._platform_info",
        _linux_wayland_without_native_tools(),
    )
    def test_screenshots_false(self) -> None:
        caps = get_capabilities()
        assert caps["screenshots"] is False

    @patch(
        "agent.desktop.platform_utils._platform_info",
        _linux_wayland_without_native_tools(),
    )
    def test_mouse_control_false(self) -> None:
        caps = get_capabilities()
        assert caps["mouse_control"] is False

    @patch(
        "agent.desktop.platform_utils._platform_info",
        _linux_wayland_without_native_tools(),
    )
    def test_keyboard_input_false(self) -> None:
        caps = get_capabilities()
        assert caps["keyboard_input"] is False

    @patch(
        "agent.desktop.platform_utils._platform_info",
        _linux_wayland_without_native_tools(),
    )
    def test_wayland_native_false_without_tools(self) -> None:
        caps = get_capabilities()
        assert caps["wayland_native"] is False

    @patch(
        "agent.desktop.platform_utils._platform_info",
        _linux_wayland_without_native_tools(),
    )
    def test_app_launching_always_true(self) -> None:
        caps = get_capabilities()
        assert caps["app_launching"] is True


class TestGetCapabilitiesLinuxWaylandPartialTools:
    """Capabilities on Wayland with only grim (no wtype/ydotool)."""

    @patch(
        "agent.desktop.platform_utils._platform_info",
        _linux_wayland_partial_tools(),
    )
    def test_screenshots_true_via_grim(self) -> None:
        caps = get_capabilities()
        assert caps["screenshots"] is True

    @patch(
        "agent.desktop.platform_utils._platform_info",
        _linux_wayland_partial_tools(),
    )
    def test_mouse_control_false_without_ydotool(self) -> None:
        caps = get_capabilities()
        assert caps["mouse_control"] is False

    @patch(
        "agent.desktop.platform_utils._platform_info",
        _linux_wayland_partial_tools(),
    )
    def test_keyboard_input_false_without_wtype(self) -> None:
        caps = get_capabilities()
        assert caps["keyboard_input"] is False

    @patch(
        "agent.desktop.platform_utils._platform_info",
        _linux_wayland_partial_tools(),
    )
    def test_wayland_native_true_with_grim_only(self) -> None:
        """wayland_native requires grim OR wtype; grim alone suffices."""
        caps = get_capabilities()
        assert caps["wayland_native"] is True


class TestGetCapabilitiesWindows:
    """Capabilities on Windows."""

    @patch(
        "agent.desktop.platform_utils._platform_info",
        _windows_with_pygetwindow(),
    )
    def test_display_is_true(self) -> None:
        caps = get_capabilities()
        assert caps["display"] is True

    @patch(
        "agent.desktop.platform_utils._platform_info",
        _windows_with_pygetwindow(),
    )
    def test_screenshots_via_pyautogui(self) -> None:
        caps = get_capabilities()
        assert caps["screenshots"] is True

    @patch(
        "agent.desktop.platform_utils._platform_info",
        _windows_with_pygetwindow(),
    )
    def test_mouse_control_via_pyautogui(self) -> None:
        caps = get_capabilities()
        assert caps["mouse_control"] is True

    @patch(
        "agent.desktop.platform_utils._platform_info",
        _windows_with_pygetwindow(),
    )
    def test_keyboard_input_via_pyautogui(self) -> None:
        caps = get_capabilities()
        assert caps["keyboard_input"] is True

    @patch(
        "agent.desktop.platform_utils._platform_info",
        _windows_with_pygetwindow(),
    )
    def test_window_management_via_pygetwindow(self) -> None:
        caps = get_capabilities()
        assert caps["window_management"] is True

    @patch(
        "agent.desktop.platform_utils._platform_info",
        _windows_with_pygetwindow(),
    )
    def test_wayland_native_false_on_windows(self) -> None:
        caps = get_capabilities()
        assert caps["wayland_native"] is False

    @patch(
        "agent.desktop.platform_utils._platform_info",
        _windows_without_pygetwindow(),
    )
    def test_window_management_false_without_pygetwindow(self) -> None:
        caps = get_capabilities()
        assert caps["window_management"] is False


class TestGetCapabilitiesMacOS:
    """Capabilities on macOS."""

    @patch("agent.desktop.platform_utils._platform_info", _macos())
    def test_display_is_true(self) -> None:
        caps = get_capabilities()
        assert caps["display"] is True

    @patch("agent.desktop.platform_utils._platform_info", _macos())
    def test_screenshots_via_pyautogui(self) -> None:
        caps = get_capabilities()
        assert caps["screenshots"] is True

    @patch("agent.desktop.platform_utils._platform_info", _macos())
    def test_window_management_via_osascript(self) -> None:
        caps = get_capabilities()
        assert caps["window_management"] is True

    @patch("agent.desktop.platform_utils._platform_info", _macos())
    def test_wayland_native_false_on_macos(self) -> None:
        caps = get_capabilities()
        assert caps["wayland_native"] is False


class TestGetCapabilitiesNoDisplay:
    """Capabilities on headless server."""

    @patch("agent.desktop.platform_utils._platform_info", _no_display())
    def test_display_is_false(self) -> None:
        caps = get_capabilities()
        assert caps["display"] is False

    @patch("agent.desktop.platform_utils._platform_info", _no_display())
    def test_screenshots_false(self) -> None:
        caps = get_capabilities()
        assert caps["screenshots"] is False

    @patch("agent.desktop.platform_utils._platform_info", _no_display())
    def test_mouse_control_false(self) -> None:
        caps = get_capabilities()
        assert caps["mouse_control"] is False

    @patch("agent.desktop.platform_utils._platform_info", _no_display())
    def test_keyboard_input_false(self) -> None:
        caps = get_capabilities()
        assert caps["keyboard_input"] is False

    @patch("agent.desktop.platform_utils._platform_info", _no_display())
    def test_window_management_false(self) -> None:
        caps = get_capabilities()
        assert caps["window_management"] is False

    @patch("agent.desktop.platform_utils._platform_info", _no_display())
    def test_app_launching_still_true(self) -> None:
        """App launching is always available via subprocess."""
        caps = get_capabilities()
        assert caps["app_launching"] is True

    @patch("agent.desktop.platform_utils._platform_info", _no_display())
    def test_wayland_native_false(self) -> None:
        caps = get_capabilities()
        assert caps["wayland_native"] is False


class TestGetCapabilitiesUnknownOS:
    """Capabilities on unknown OS."""

    @patch("agent.desktop.platform_utils._platform_info", _unknown_os())
    def test_all_false_except_app_launching(self) -> None:
        caps = get_capabilities()
        assert caps["display"] is False
        assert caps["screenshots"] is False
        assert caps["mouse_control"] is False
        assert caps["keyboard_input"] is False
        assert caps["window_management"] is False
        assert caps["app_launching"] is True
        assert caps["wayland_native"] is False


# ---------------------------------------------------------------------------
# get_capabilities_summary() tests
# ---------------------------------------------------------------------------

class TestGetCapabilitiesSummaryLinuxX11:
    """Summary output for Linux X11."""

    @patch("agent.desktop.platform_utils._platform_info", _linux_x11_with_pyautogui())
    def test_contains_os_and_display(self) -> None:
        summary = get_capabilities_summary()
        assert "OS: linux" in summary
        assert "Display: x11" in summary

    @patch("agent.desktop.platform_utils._platform_info", _linux_x11_with_pyautogui())
    def test_contains_screen_info(self) -> None:
        summary = get_capabilities_summary()
        assert "1920x1080" in summary
        assert "scale: 1.0x" in summary
        assert "monitors: 2" in summary

    @patch("agent.desktop.platform_utils._platform_info", _linux_x11_with_pyautogui())
    def test_lists_available_capabilities(self) -> None:
        summary = get_capabilities_summary()
        assert "Available:" in summary
        assert "display" in summary
        assert "screenshots" in summary
        assert "app_launching" in summary

    @patch("agent.desktop.platform_utils._platform_info", _linux_x11_with_pyautogui())
    def test_lists_unavailable_wayland_native(self) -> None:
        summary = get_capabilities_summary()
        assert "Unavailable:" in summary
        assert "wayland_native" in summary

    @patch("agent.desktop.platform_utils._platform_info", _linux_x11_with_pyautogui())
    def test_no_wayland_tools_section(self) -> None:
        """X11 should not have Wayland tools section."""
        summary = get_capabilities_summary()
        assert "Wayland tools:" not in summary


class TestGetCapabilitiesSummaryWaylandWithTools:
    """Summary output for Linux Wayland with native tools."""

    @patch(
        "agent.desktop.platform_utils._platform_info",
        _linux_wayland_with_native_tools(),
    )
    def test_contains_wayland_display(self) -> None:
        summary = get_capabilities_summary()
        assert "Display: wayland" in summary

    @patch(
        "agent.desktop.platform_utils._platform_info",
        _linux_wayland_with_native_tools(),
    )
    def test_contains_screen_info(self) -> None:
        summary = get_capabilities_summary()
        assert "2560x1440" in summary
        assert "scale: 1.5x" in summary

    @patch(
        "agent.desktop.platform_utils._platform_info",
        _linux_wayland_with_native_tools(),
    )
    def test_lists_wayland_tools(self) -> None:
        summary = get_capabilities_summary()
        assert "Wayland tools:" in summary
        assert "grim (screenshots)" in summary
        assert "wtype (keyboard)" in summary
        assert "ydotool (mouse)" in summary
        assert "slurp (region select)" in summary

    @patch(
        "agent.desktop.platform_utils._platform_info",
        _linux_wayland_with_native_tools(),
    )
    def test_wayland_native_in_available(self) -> None:
        summary = get_capabilities_summary()
        assert "wayland_native" in summary
        # It should be in the Available line, not in Unavailable
        lines = summary.split("\n")
        available_line = [ln for ln in lines if ln.startswith("Available:")][0]
        assert "wayland_native" in available_line


class TestGetCapabilitiesSummaryWaylandNoTools:
    """Summary output for Linux Wayland without native tools."""

    @patch(
        "agent.desktop.platform_utils._platform_info",
        _linux_wayland_without_native_tools(),
    )
    def test_shows_no_native_tools_note(self) -> None:
        summary = get_capabilities_summary()
        assert "Wayland detected but no native tools found" in summary
        assert "Install grim, wtype, ydotool" in summary

    @patch(
        "agent.desktop.platform_utils._platform_info",
        _linux_wayland_without_native_tools(),
    )
    def test_does_not_show_wayland_tools_list(self) -> None:
        summary = get_capabilities_summary()
        assert "Wayland tools:" not in summary


class TestGetCapabilitiesSummaryWaylandPartialTools:
    """Summary output for Wayland with only grim."""

    @patch(
        "agent.desktop.platform_utils._platform_info",
        _linux_wayland_partial_tools(),
    )
    def test_lists_only_grim(self) -> None:
        summary = get_capabilities_summary()
        assert "Wayland tools:" in summary
        assert "grim (screenshots)" in summary
        assert "wtype" not in summary
        assert "ydotool" not in summary
        assert "slurp" not in summary


class TestGetCapabilitiesSummaryWindows:
    """Summary output for Windows."""

    @patch(
        "agent.desktop.platform_utils._platform_info",
        _windows_with_pygetwindow(),
    )
    def test_contains_os_and_display(self) -> None:
        summary = get_capabilities_summary()
        assert "OS: windows" in summary
        assert "Display: win32" in summary

    @patch(
        "agent.desktop.platform_utils._platform_info",
        _windows_with_pygetwindow(),
    )
    def test_contains_screen_info(self) -> None:
        summary = get_capabilities_summary()
        assert "2560x1440" in summary
        assert "scale: 1.5x" in summary
        assert "monitors: 3" in summary

    @patch(
        "agent.desktop.platform_utils._platform_info",
        _windows_with_pygetwindow(),
    )
    def test_no_wayland_section(self) -> None:
        summary = get_capabilities_summary()
        assert "Wayland" not in summary


class TestGetCapabilitiesSummaryMacOS:
    """Summary output for macOS."""

    @patch("agent.desktop.platform_utils._platform_info", _macos())
    def test_contains_os_and_display(self) -> None:
        summary = get_capabilities_summary()
        assert "OS: macos" in summary
        assert "Display: quartz" in summary

    @patch("agent.desktop.platform_utils._platform_info", _macos())
    def test_contains_retina_scale(self) -> None:
        summary = get_capabilities_summary()
        assert "scale: 2.0x" in summary

    @patch("agent.desktop.platform_utils._platform_info", _macos())
    def test_window_management_available(self) -> None:
        summary = get_capabilities_summary()
        lines = summary.split("\n")
        available_line = [ln for ln in lines if ln.startswith("Available:")][0]
        assert "window_management" in available_line


class TestGetCapabilitiesSummaryNoDisplay:
    """Summary output for headless server."""

    @patch("agent.desktop.platform_utils._platform_info", _no_display())
    def test_display_shows_none(self) -> None:
        summary = get_capabilities_summary()
        assert "Display: none" in summary

    @patch("agent.desktop.platform_utils._platform_info", _no_display())
    def test_most_capabilities_unavailable(self) -> None:
        summary = get_capabilities_summary()
        assert "Unavailable:" in summary
        # display, screenshots, mouse_control, keyboard_input, window_management,
        # wayland_native should all be unavailable
        lines = summary.split("\n")
        unavailable_line = [ln for ln in lines if ln.startswith("Unavailable:")][0]
        assert "display" in unavailable_line
        assert "screenshots" in unavailable_line

    @patch("agent.desktop.platform_utils._platform_info", _no_display())
    def test_app_launching_still_available(self) -> None:
        summary = get_capabilities_summary()
        lines = summary.split("\n")
        available_line = [ln for ln in lines if ln.startswith("Available:")][0]
        assert "app_launching" in available_line

    @patch("agent.desktop.platform_utils._platform_info", _no_display())
    def test_summary_is_multiline_string(self) -> None:
        summary = get_capabilities_summary()
        assert isinstance(summary, str)
        lines = summary.strip().split("\n")
        assert len(lines) >= 3  # OS line, Screen line, Available/Unavailable


class TestGetCapabilitiesReturnType:
    """Verify return types and structure."""

    @patch("agent.desktop.platform_utils._platform_info", _linux_x11_with_pyautogui())
    def test_returns_dict_of_str_bool(self) -> None:
        caps = get_capabilities()
        assert isinstance(caps, dict)
        for key, value in caps.items():
            assert isinstance(key, str), f"Key {key!r} is not str"
            assert isinstance(value, bool), f"Value for {key!r} is not bool"

    @patch("agent.desktop.platform_utils._platform_info", _linux_x11_with_pyautogui())
    def test_summary_returns_str(self) -> None:
        summary = get_capabilities_summary()
        assert isinstance(summary, str)
        assert len(summary) > 0


# ---------------------------------------------------------------------------
# Accessibility tree capability tests
# ---------------------------------------------------------------------------

class TestAccessibilityTreeCapability:
    """Tests for the accessibility_tree capability key."""

    @patch(
        "agent.desktop.platform_utils._platform_info",
        _windows_with_pygetwindow(),
    )
    def test_windows_with_uiautomation(self) -> None:
        """Windows with uiautomation should have accessibility_tree=True."""
        caps = get_capabilities()
        assert caps["accessibility_tree"] is True

    @patch(
        "agent.desktop.platform_utils._platform_info",
        _windows_without_pygetwindow(),
    )
    def test_windows_without_uiautomation(self) -> None:
        """Windows without uiautomation should have accessibility_tree=False."""
        caps = get_capabilities()
        assert caps["accessibility_tree"] is False

    @patch("agent.desktop.platform_utils._platform_info", _macos())
    def test_macos_with_osascript(self) -> None:
        """macOS with osascript should have accessibility_tree=True."""
        caps = get_capabilities()
        assert caps["accessibility_tree"] is True

    @patch("agent.desktop.platform_utils._platform_info", _no_display())
    def test_headless_no_accessibility(self) -> None:
        """Headless server should have accessibility_tree=False."""
        caps = get_capabilities()
        assert caps["accessibility_tree"] is False


class TestAccessibilityTreeSummary:
    """Tests for accessibility info in capabilities summary."""

    @patch(
        "agent.desktop.platform_utils._platform_info",
        _windows_with_pygetwindow(),
    )
    def test_windows_summary_shows_uiautomation(self) -> None:
        summary = get_capabilities_summary()
        assert "uiautomation" in summary
        assert "SoM" in summary

    @patch("agent.desktop.platform_utils._platform_info", _macos())
    def test_macos_summary_shows_applescript(self) -> None:
        summary = get_capabilities_summary()
        assert "AppleScript" in summary
        assert "SoM" in summary

    @patch(
        "agent.desktop.platform_utils._platform_info",
        _windows_without_pygetwindow(),
    )
    def test_windows_no_uiautomation_shows_install_hint(self) -> None:
        summary = get_capabilities_summary()
        assert "not available" in summary
        assert "uiautomation" in summary
