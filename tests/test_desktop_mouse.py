"""Tests for desktop mouse control."""

from __future__ import annotations

from collections import namedtuple
from unittest.mock import MagicMock, patch

from agent.desktop.mouse import (
    _validate_coords,
    click,
    drag,
    get_position,
    move_to,
    scroll,
)
from agent.desktop.platform_utils import OSType, PlatformInfo

_PLATFORM = PlatformInfo(
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
)

Position = namedtuple("Position", ["x", "y"])


class TestValidateCoords:
    """Tests for coordinate validation."""

    @patch("agent.desktop.mouse.get_platform", return_value=_PLATFORM)
    def test_clamps_negative_to_zero(self, _: object) -> None:
        x, y = _validate_coords(-10, -20)
        assert x == 0
        assert y == 0

    @patch("agent.desktop.mouse.get_platform", return_value=_PLATFORM)
    def test_clamps_over_max(self, _: object) -> None:
        x, y = _validate_coords(5000, 3000)
        assert x == 1919
        assert y == 1079

    @patch("agent.desktop.mouse.get_platform", return_value=_PLATFORM)
    def test_passes_valid_coords(self, _: object) -> None:
        x, y = _validate_coords(500, 300)
        assert x == 500
        assert y == 300


class TestMoveTo:
    """Tests for move_to()."""

    @patch("agent.desktop.mouse.get_platform", return_value=_PLATFORM)
    async def test_move_calls_pyautogui(self, _: object) -> None:
        mock_pyautogui = MagicMock()

        with patch.dict("sys.modules", {"pyautogui": mock_pyautogui}):
            result = await move_to(100, 200, duration=0.0)
            assert result == (100, 200)
            mock_pyautogui.moveTo.assert_called_once_with(100, 200, duration=0.0)


class TestClick:
    """Tests for click()."""

    @patch("agent.desktop.mouse.get_platform", return_value=_PLATFORM)
    async def test_click_at_coords(self, _: object) -> None:
        mock_pyautogui = MagicMock()
        mock_pyautogui.position.return_value = Position(100, 200)

        with patch.dict("sys.modules", {"pyautogui": mock_pyautogui}):
            result = await click(x=100, y=200, button="left", clicks=1)
            assert "Clicked left at (100, 200)" in result
            mock_pyautogui.click.assert_called_once_with(x=100, y=200, button="left", clicks=1)

    @patch("agent.desktop.mouse.get_platform", return_value=_PLATFORM)
    async def test_double_click(self, _: object) -> None:
        mock_pyautogui = MagicMock()
        mock_pyautogui.position.return_value = Position(50, 60)

        with patch.dict("sys.modules", {"pyautogui": mock_pyautogui}):
            result = await click(x=50, y=60, clicks=2)
            assert "x2" in result
            mock_pyautogui.click.assert_called_once_with(x=50, y=60, button="left", clicks=2)

    @patch("agent.desktop.mouse.get_platform", return_value=_PLATFORM)
    async def test_click_current_position(self, _: object) -> None:
        mock_pyautogui = MagicMock()
        mock_pyautogui.position.return_value = Position(300, 400)

        with patch.dict("sys.modules", {"pyautogui": mock_pyautogui}):
            result = await click()
            assert "300" in result
            mock_pyautogui.click.assert_called_once_with(x=None, y=None, button="left", clicks=1)


class TestDrag:
    """Tests for drag()."""

    @patch("agent.desktop.mouse.get_platform", return_value=_PLATFORM)
    async def test_drag_calls_pyautogui(self, _: object) -> None:
        mock_pyautogui = MagicMock()

        with patch.dict("sys.modules", {"pyautogui": mock_pyautogui}):
            result = await drag(100, 100, 300, 300, duration=0.0)
            assert "Dragged from (100, 100) to (300, 300)" in result
            mock_pyautogui.moveTo.assert_called_once_with(100, 100, duration=0.1)
            mock_pyautogui.drag.assert_called_once_with(200, 200, duration=0.0, button="left")


class TestScroll:
    """Tests for scroll()."""

    @patch("agent.desktop.mouse.get_platform", return_value=_PLATFORM)
    async def test_scroll_up(self, _: object) -> None:
        mock_pyautogui = MagicMock()

        with patch.dict("sys.modules", {"pyautogui": mock_pyautogui}):
            result = await scroll(3)
            assert "up" in result
            assert "3" in result
            mock_pyautogui.scroll.assert_called_once_with(3)

    @patch("agent.desktop.mouse.get_platform", return_value=_PLATFORM)
    async def test_scroll_down(self, _: object) -> None:
        mock_pyautogui = MagicMock()

        with patch.dict("sys.modules", {"pyautogui": mock_pyautogui}):
            result = await scroll(-5)
            assert "down" in result
            assert "5" in result

    @patch("agent.desktop.mouse.get_platform", return_value=_PLATFORM)
    async def test_scroll_at_position(self, _: object) -> None:
        mock_pyautogui = MagicMock()

        with patch.dict("sys.modules", {"pyautogui": mock_pyautogui}):
            result = await scroll(2, x=100, y=200)
            assert "up" in result
            mock_pyautogui.scroll.assert_called_once_with(2, x=100, y=200)


class TestGetPosition:
    """Tests for get_position()."""

    @patch("agent.desktop.mouse.get_platform", return_value=_PLATFORM)
    async def test_returns_tuple(self, _: object) -> None:
        mock_pyautogui = MagicMock()
        mock_pyautogui.position.return_value = Position(555, 666)

        with patch.dict("sys.modules", {"pyautogui": mock_pyautogui}):
            x, y = await get_position()
            assert x == 555
            assert y == 666
