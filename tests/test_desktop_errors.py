"""Tests for desktop error handling module."""

from __future__ import annotations

from unittest.mock import patch

import pytest

from agent.desktop.errors import (
    DesktopError,
    MissingDependencyError,
    NoDisplayError,
    PlatformNotSupportedError,
    desktop_op,
)
from agent.desktop.platform_utils import OSType, PlatformInfo


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_platform(
    has_display: bool = True,
    os_type: OSType = OSType.LINUX,
    display_server: str = "x11",
) -> PlatformInfo:
    """Build a PlatformInfo with sensible defaults for testing."""
    return PlatformInfo(
        os_type=os_type,
        has_display=has_display,
        display_server=display_server if has_display else "",
        has_pyautogui=True,
        has_wmctrl=False,
        has_xdotool=False,
        has_osascript=False,
        screen_width=1920,
        screen_height=1080,
        scale_factor=1.0,
    )


# ---------------------------------------------------------------------------
# Exception hierarchy tests
# ---------------------------------------------------------------------------

class TestExceptionHierarchy:
    """Verify the custom desktop exception class hierarchy."""

    def test_desktop_error_is_exception(self) -> None:
        assert issubclass(DesktopError, Exception)

    def test_platform_not_supported_is_desktop_error(self) -> None:
        assert issubclass(PlatformNotSupportedError, DesktopError)

    def test_missing_dependency_is_desktop_error(self) -> None:
        assert issubclass(MissingDependencyError, DesktopError)

    def test_no_display_is_desktop_error(self) -> None:
        assert issubclass(NoDisplayError, DesktopError)

    def test_desktop_error_can_be_raised_and_caught(self) -> None:
        with pytest.raises(DesktopError, match="something went wrong"):
            raise DesktopError("something went wrong")

    def test_platform_not_supported_carries_message(self) -> None:
        err = PlatformNotSupportedError("FreeBSD is not supported")
        assert str(err) == "FreeBSD is not supported"

    def test_missing_dependency_carries_message(self) -> None:
        err = MissingDependencyError("pyautogui is required")
        assert str(err) == "pyautogui is required"

    def test_no_display_carries_message(self) -> None:
        err = NoDisplayError("DISPLAY not set")
        assert str(err) == "DISPLAY not set"

    def test_catch_subclass_via_base(self) -> None:
        """All subclasses should be catchable via DesktopError."""
        for cls in (PlatformNotSupportedError, MissingDependencyError, NoDisplayError):
            with pytest.raises(DesktopError):
                raise cls("test")


# ---------------------------------------------------------------------------
# @desktop_op decorator tests
# ---------------------------------------------------------------------------

class TestDesktopOpNoDisplay:
    """When has_display=False the decorator short-circuits."""

    @patch("agent.desktop.platform_utils.get_platform")
    async def test_returns_unavailable_when_no_display(self, mock_gp: object) -> None:
        mock_gp.return_value = _make_platform(has_display=False)  # type: ignore[union-attr]

        @desktop_op("Screenshot")
        async def take_screenshot() -> str:
            return "should not reach here"

        result = await take_screenshot()
        assert result.startswith("[UNAVAILABLE] Screenshot:")
        assert "No display available" in result
        assert "GUI environment" in result

    @patch("agent.desktop.platform_utils.get_platform")
    async def test_inner_function_never_called_when_no_display(
        self, mock_gp: object
    ) -> None:
        mock_gp.return_value = _make_platform(has_display=False)  # type: ignore[union-attr]
        called = False

        @desktop_op("Mouse")
        async def move_mouse() -> str:
            nonlocal called
            called = True
            return "moved"

        await move_mouse()
        assert not called

    @patch("agent.desktop.platform_utils.get_platform")
    async def test_feature_name_appears_in_message(self, mock_gp: object) -> None:
        mock_gp.return_value = _make_platform(has_display=False)  # type: ignore[union-attr]

        @desktop_op("Keyboard Input")
        async def type_text() -> str:
            return "typed"

        result = await type_text()
        assert "[UNAVAILABLE] Keyboard Input:" in result


class TestDesktopOpImportError:
    """ImportError inside the wrapped function produces a consistent message."""

    @patch("agent.desktop.platform_utils.get_platform")
    async def test_catches_import_error(self, mock_gp: object) -> None:
        mock_gp.return_value = _make_platform(has_display=True)  # type: ignore[union-attr]

        @desktop_op("Screenshot")
        async def take_screenshot() -> str:
            raise ImportError("No module named 'pyautogui'")

        result = await take_screenshot()
        assert result.startswith("[UNAVAILABLE] Screenshot:")
        assert "Missing dependency" in result
        assert "pyautogui" in result

    @patch("agent.desktop.platform_utils.get_platform")
    async def test_import_error_includes_dash_separator(self, mock_gp: object) -> None:
        mock_gp.return_value = _make_platform(has_display=True)  # type: ignore[union-attr]

        @desktop_op("Browser")
        async def launch_browser() -> str:
            raise ImportError("playwright not installed")

        result = await launch_browser()
        # The format is "[UNAVAILABLE] feature: Missing dependency — <error>"
        assert "Missing dependency \u2014" in result

    @patch("agent.desktop.platform_utils.get_platform")
    async def test_import_error_preserves_original_message(self, mock_gp: object) -> None:
        mock_gp.return_value = _make_platform(has_display=True)  # type: ignore[union-attr]
        original_msg = "No module named 'Xlib'"

        @desktop_op("Window Manager")
        async def manage_windows() -> str:
            raise ImportError(original_msg)

        result = await manage_windows()
        assert original_msg in result


class TestDesktopOpPlatformNotSupported:
    """PlatformNotSupportedError is caught and formatted."""

    @patch("agent.desktop.platform_utils.get_platform")
    async def test_catches_platform_not_supported(self, mock_gp: object) -> None:
        mock_gp.return_value = _make_platform(has_display=True)  # type: ignore[union-attr]

        @desktop_op("Desktop Control")
        async def do_desktop() -> str:
            raise PlatformNotSupportedError("FreeBSD is not supported")

        result = await do_desktop()
        assert result == "[UNAVAILABLE] Desktop Control: FreeBSD is not supported"

    @patch("agent.desktop.platform_utils.get_platform")
    async def test_catches_missing_dependency_error(self, mock_gp: object) -> None:
        mock_gp.return_value = _make_platform(has_display=True)  # type: ignore[union-attr]

        @desktop_op("Screen Capture")
        async def capture() -> str:
            raise MissingDependencyError("grim is required for Wayland screenshots")

        result = await capture()
        assert result == (
            "[UNAVAILABLE] Screen Capture: grim is required for Wayland screenshots"
        )

    @patch("agent.desktop.platform_utils.get_platform")
    async def test_catches_no_display_error(self, mock_gp: object) -> None:
        mock_gp.return_value = _make_platform(has_display=True)  # type: ignore[union-attr]

        @desktop_op("Window Listing")
        async def list_windows() -> str:
            raise NoDisplayError("Display connection lost")

        result = await list_windows()
        assert result == "[UNAVAILABLE] Window Listing: Display connection lost"


class TestDesktopOpGenericException:
    """Unrecognised exceptions produce [ERROR] messages."""

    @patch("agent.desktop.platform_utils.get_platform")
    async def test_catches_runtime_error(self, mock_gp: object) -> None:
        mock_gp.return_value = _make_platform(has_display=True)  # type: ignore[union-attr]

        @desktop_op("Mouse Move")
        async def move() -> str:
            raise RuntimeError("coordinates out of range")

        result = await move()
        assert result.startswith("[ERROR] Mouse Move:")
        assert "coordinates out of range" in result

    @patch("agent.desktop.platform_utils.get_platform")
    async def test_catches_value_error(self, mock_gp: object) -> None:
        mock_gp.return_value = _make_platform(has_display=True)  # type: ignore[union-attr]

        @desktop_op("Keyboard")
        async def type_key() -> str:
            raise ValueError("invalid key name")

        result = await type_key()
        assert result == "[ERROR] Keyboard: invalid key name"

    @patch("agent.desktop.platform_utils.get_platform")
    async def test_catches_os_error(self, mock_gp: object) -> None:
        mock_gp.return_value = _make_platform(has_display=True)  # type: ignore[union-attr]

        @desktop_op("File Save")
        async def save_file() -> str:
            raise OSError("permission denied")

        result = await save_file()
        assert "[ERROR] File Save:" in result
        assert "permission denied" in result


class TestDesktopOpPassthrough:
    """When everything works, the decorated function returns normally."""

    @patch("agent.desktop.platform_utils.get_platform")
    async def test_returns_result_on_success(self, mock_gp: object) -> None:
        mock_gp.return_value = _make_platform(has_display=True)  # type: ignore[union-attr]

        @desktop_op("Screenshot")
        async def take_screenshot() -> str:
            return "screenshot_data_here"

        result = await take_screenshot()
        assert result == "screenshot_data_here"

    @patch("agent.desktop.platform_utils.get_platform")
    async def test_passes_args_and_kwargs(self, mock_gp: object) -> None:
        mock_gp.return_value = _make_platform(has_display=True)  # type: ignore[union-attr]

        @desktop_op("Mouse Click")
        async def click(x: int, y: int, button: str = "left") -> str:
            return f"clicked {x},{y} {button}"

        result = await click(100, 200, button="right")
        assert result == "clicked 100,200 right"

    @patch("agent.desktop.platform_utils.get_platform")
    async def test_returns_non_string_value(self, mock_gp: object) -> None:
        mock_gp.return_value = _make_platform(has_display=True)  # type: ignore[union-attr]

        @desktop_op("Get Info")
        async def get_info() -> dict:
            return {"width": 1920, "height": 1080}

        result = await get_info()
        assert result == {"width": 1920, "height": 1080}

    @patch("agent.desktop.platform_utils.get_platform")
    async def test_returns_none(self, mock_gp: object) -> None:
        mock_gp.return_value = _make_platform(has_display=True)  # type: ignore[union-attr]

        @desktop_op("Side Effect")
        async def do_something() -> None:
            pass

        result = await do_something()
        assert result is None


class TestDesktopOpDecoratorProperties:
    """Verify decorator preserves function metadata."""

    def test_preserves_function_name(self) -> None:
        @desktop_op("Test")
        async def my_function() -> str:
            return "ok"

        assert my_function.__name__ == "my_function"

    def test_preserves_docstring(self) -> None:
        @desktop_op("Test")
        async def documented_function() -> str:
            """This function does something."""
            return "ok"

        assert documented_function.__doc__ == "This function does something."

    def test_preserves_module(self) -> None:
        @desktop_op("Test")
        async def my_func() -> str:
            return "ok"

        assert my_func.__module__ == __name__


class TestDesktopOpExceptionPriority:
    """Verify that more specific exceptions are caught before generic ones."""

    @patch("agent.desktop.platform_utils.get_platform")
    async def test_import_error_takes_precedence_over_generic(
        self, mock_gp: object
    ) -> None:
        """ImportError is a subclass of Exception but has its own handler."""
        mock_gp.return_value = _make_platform(has_display=True)  # type: ignore[union-attr]

        @desktop_op("Feature")
        async def feat() -> str:
            raise ImportError("missing lib")

        result = await feat()
        # Should be caught by ImportError handler, not generic
        assert "[UNAVAILABLE]" in result
        assert "Missing dependency" in result
        assert "[ERROR]" not in result

    @patch("agent.desktop.platform_utils.get_platform")
    async def test_platform_error_takes_precedence_over_generic(
        self, mock_gp: object
    ) -> None:
        """DesktopError subclasses have their own handler."""
        mock_gp.return_value = _make_platform(has_display=True)  # type: ignore[union-attr]

        @desktop_op("Feature")
        async def feat() -> str:
            raise PlatformNotSupportedError("not supported")

        result = await feat()
        assert "[UNAVAILABLE]" in result
        assert "[ERROR]" not in result

    @patch("agent.desktop.platform_utils.get_platform")
    async def test_no_display_check_takes_precedence_over_everything(
        self, mock_gp: object
    ) -> None:
        """has_display=False short-circuits before the function even runs."""
        mock_gp.return_value = _make_platform(has_display=False)  # type: ignore[union-attr]

        @desktop_op("Feature")
        async def feat() -> str:
            raise RuntimeError("this should never execute")

        result = await feat()
        assert "No display available" in result
