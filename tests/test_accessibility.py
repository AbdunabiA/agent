"""Tests for cross-platform accessibility tree extraction."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from agent.desktop.accessibility import (
    UIElement,
    _filter_elements,
    _normalize_role,
    _parse_macos_output,
    _sort_and_number,
    get_ui_elements,
)

# ---------------------------------------------------------------------------
# TestUIElementDataclass
# ---------------------------------------------------------------------------

class TestUIElementDataclass:
    """UIElement creation and defaults."""

    def test_create_with_all_fields(self) -> None:
        el = UIElement(
            id=1, role="button", name="OK", x=10, y=20, width=80, height=30,
            is_enabled=True, is_focused=True, value="",
        )
        assert el.id == 1
        assert el.role == "button"
        assert el.name == "OK"
        assert el.x == 10
        assert el.y == 20
        assert el.width == 80
        assert el.height == 30
        assert el.is_enabled is True
        assert el.is_focused is True
        assert el.value == ""

    def test_defaults(self) -> None:
        el = UIElement(id=1, role="button", name="X", x=0, y=0, width=10, height=10)
        assert el.is_enabled is True
        assert el.is_focused is False
        assert el.value == ""

    def test_with_value(self) -> None:
        el = UIElement(id=2, role="text_field", name="Search", x=0, y=0,
                       width=200, height=30, value="hello")
        assert el.value == "hello"


# ---------------------------------------------------------------------------
# TestNormalizeRole
# ---------------------------------------------------------------------------

class TestNormalizeRole:
    """Role name normalization across platforms."""

    def test_windows_button_control(self) -> None:
        assert _normalize_role("ButtonControl") == "button"

    def test_windows_edit_control(self) -> None:
        assert _normalize_role("EditControl") == "text_field"

    def test_windows_menuitem_control(self) -> None:
        assert _normalize_role("MenuItemControl") == "menu_item"

    def test_windows_checkbox_control(self) -> None:
        assert _normalize_role("CheckBoxControl") == "checkbox"

    def test_windows_combobox_control(self) -> None:
        assert _normalize_role("ComboBoxControl") == "dropdown"

    def test_macos_ax_button(self) -> None:
        assert _normalize_role("AXButton") == "button"

    def test_macos_ax_text_field(self) -> None:
        assert _normalize_role("AXTextField") == "text_field"

    def test_macos_ax_menu_item(self) -> None:
        assert _normalize_role("AXMenuItem") == "menu_item"

    def test_macos_ax_popup_button(self) -> None:
        assert _normalize_role("AXPopUpButton") == "dropdown"

    def test_macos_ax_link(self) -> None:
        assert _normalize_role("AXLink") == "link"

    def test_linux_push_button(self) -> None:
        assert _normalize_role("push button") == "button"

    def test_linux_toggle_button(self) -> None:
        assert _normalize_role("toggle button") == "toggle"

    def test_linux_entry(self) -> None:
        assert _normalize_role("entry") == "text_field"

    def test_linux_menu_item(self) -> None:
        assert _normalize_role("menu item") == "menu_item"

    def test_linux_check_box(self) -> None:
        assert _normalize_role("check box") == "checkbox"

    def test_linux_combo_box(self) -> None:
        assert _normalize_role("combo box") == "dropdown"

    def test_unknown_role_passthrough(self) -> None:
        assert _normalize_role("SomeNewRole") == "somenewrole"

    def test_case_insensitive_match(self) -> None:
        assert _normalize_role("BUTTON") == "button"
        assert _normalize_role("Slider") == "slider"

    def test_empty_string(self) -> None:
        assert _normalize_role("") == ""


# ---------------------------------------------------------------------------
# TestFilterElements
# ---------------------------------------------------------------------------

class TestFilterElements:
    """Element filtering removes invalid/non-interactive elements."""

    def _make_el(self, role: str = "button", x: int = 100, y: int = 100,
                 w: int = 80, h: int = 30, enabled: bool = True,
                 name: str = "Test") -> UIElement:
        return UIElement(id=0, role=role, name=name, x=x, y=y,
                         width=w, height=h, is_enabled=enabled)

    def test_keeps_interactive_elements(self) -> None:
        elements = [self._make_el("button"), self._make_el("text_field")]
        filtered = _filter_elements(elements, 1920, 1080)
        assert len(filtered) == 2

    def test_removes_non_interactive_roles(self) -> None:
        elements = [
            self._make_el("button"),
            self._make_el("panel"),      # non-interactive
            self._make_el("separator"),  # non-interactive
        ]
        filtered = _filter_elements(elements, 1920, 1080)
        assert len(filtered) == 1
        assert filtered[0].role == "button"

    def test_removes_zero_width(self) -> None:
        elements = [self._make_el(w=0)]
        filtered = _filter_elements(elements, 1920, 1080)
        assert len(filtered) == 0

    def test_removes_zero_height(self) -> None:
        elements = [self._make_el(h=0)]
        filtered = _filter_elements(elements, 1920, 1080)
        assert len(filtered) == 0

    def test_removes_negative_size(self) -> None:
        elements = [self._make_el(w=-10, h=-5)]
        filtered = _filter_elements(elements, 1920, 1080)
        assert len(filtered) == 0

    def test_removes_off_screen_right(self) -> None:
        elements = [self._make_el(x=1920, w=80)]
        filtered = _filter_elements(elements, 1920, 1080)
        assert len(filtered) == 0

    def test_removes_off_screen_bottom(self) -> None:
        elements = [self._make_el(y=1080, h=30)]
        filtered = _filter_elements(elements, 1920, 1080)
        assert len(filtered) == 0

    def test_removes_off_screen_left(self) -> None:
        elements = [self._make_el(x=-100, w=50)]  # x + w = -50 <= 0
        filtered = _filter_elements(elements, 1920, 1080)
        assert len(filtered) == 0

    def test_removes_off_screen_top(self) -> None:
        elements = [self._make_el(y=-50, h=30)]  # y + h = -20 <= 0
        filtered = _filter_elements(elements, 1920, 1080)
        assert len(filtered) == 0

    def test_keeps_partially_on_screen(self) -> None:
        # Element overlaps left edge but is partially visible
        elements = [self._make_el(x=-30, w=80)]  # x + w = 50 > 0
        filtered = _filter_elements(elements, 1920, 1080)
        assert len(filtered) == 1

    def test_removes_disabled_nameless(self) -> None:
        elements = [self._make_el(enabled=False, name="")]
        filtered = _filter_elements(elements, 1920, 1080)
        assert len(filtered) == 0

    def test_keeps_disabled_with_name(self) -> None:
        elements = [self._make_el(enabled=False, name="Save")]
        filtered = _filter_elements(elements, 1920, 1080)
        assert len(filtered) == 1

    def test_caps_at_max_elements(self) -> None:
        elements = [self._make_el(x=i, y=i) for i in range(300)]
        filtered = _filter_elements(elements, 1920, 1080)
        assert len(filtered) <= 200

    def test_empty_input(self) -> None:
        filtered = _filter_elements([], 1920, 1080)
        assert filtered == []


# ---------------------------------------------------------------------------
# TestSortAndNumber
# ---------------------------------------------------------------------------

class TestSortAndNumber:
    """Element sorting and ID assignment."""

    def _make_el(self, x: int, y: int) -> UIElement:
        return UIElement(id=0, role="button", name="B", x=x, y=y, width=80, height=30)

    def test_sorts_top_to_bottom(self) -> None:
        elements = [self._make_el(100, 300), self._make_el(100, 100), self._make_el(100, 200)]
        result = _sort_and_number(elements)
        assert result[0].y == 100
        assert result[1].y == 200
        assert result[2].y == 300

    def test_sorts_left_to_right_on_same_row(self) -> None:
        elements = [self._make_el(300, 100), self._make_el(100, 100), self._make_el(200, 100)]
        result = _sort_and_number(elements)
        assert result[0].x == 100
        assert result[1].x == 200
        assert result[2].x == 300

    def test_assigns_sequential_ids(self) -> None:
        elements = [self._make_el(0, 0), self._make_el(100, 0), self._make_el(0, 100)]
        result = _sort_and_number(elements)
        assert [el.id for el in result] == [1, 2, 3]

    def test_ids_start_at_1(self) -> None:
        elements = [self._make_el(50, 50)]
        result = _sort_and_number(elements)
        assert result[0].id == 1

    def test_empty_list(self) -> None:
        result = _sort_and_number([])
        assert result == []


# ---------------------------------------------------------------------------
# TestParseMacOSOutput
# ---------------------------------------------------------------------------

class TestParseMacOSOutput:
    """Parsing delimited AppleScript output."""

    def test_parses_valid_line(self) -> None:
        output = "AXButton|||OK|||100|||200|||80|||30|||true|||false\n"
        elements = _parse_macos_output(output)
        assert len(elements) == 1
        assert elements[0].role == "button"
        assert elements[0].name == "OK"
        assert elements[0].x == 100
        assert elements[0].y == 200
        assert elements[0].width == 80
        assert elements[0].height == 30

    def test_parses_multiple_lines(self) -> None:
        output = (
            "AXButton|||OK|||100|||200|||80|||30|||true|||false\n"
            "AXTextField|||Search|||10|||50|||200|||25|||true|||true\n"
        )
        elements = _parse_macos_output(output)
        assert len(elements) == 2

    def test_skips_short_lines(self) -> None:
        output = "AXButton|||OK|||100\n"  # Only 3 parts, need 6
        elements = _parse_macos_output(output)
        assert len(elements) == 0

    def test_skips_empty_lines(self) -> None:
        output = "\n\n\n"
        elements = _parse_macos_output(output)
        assert len(elements) == 0

    def test_handles_float_coordinates(self) -> None:
        output = "AXButton|||OK|||100.5|||200.7|||80.0|||30.0|||true|||false\n"
        elements = _parse_macos_output(output)
        assert len(elements) == 1
        assert elements[0].x == 100
        assert elements[0].y == 200

    def test_empty_output(self) -> None:
        elements = _parse_macos_output("")
        assert elements == []


# ---------------------------------------------------------------------------
# TestGetUIElementsWindows
# ---------------------------------------------------------------------------

class TestGetUIElementsWindows:
    """Windows accessibility extraction with mocked uiautomation."""

    @pytest.fixture
    def mock_platform_windows(self) -> PlatformInfo:
        from agent.desktop.platform_utils import OSType, PlatformInfo
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
            has_pygetwindow=True,
            has_uiautomation=True,
            has_pyatspi=False,
        )

    @patch("agent.desktop.accessibility.get_platform")
    @patch("agent.desktop.accessibility._get_elements_windows")
    async def test_calls_windows_backend(
        self, mock_get_elements: AsyncMock, mock_platform: MagicMock,
        mock_platform_windows: PlatformInfo,
    ) -> None:
        mock_platform.return_value = mock_platform_windows
        mock_get_elements.return_value = [
            UIElement(id=0, role="button", name="OK", x=100, y=200, width=80, height=30),
        ]

        result = await get_ui_elements()
        mock_get_elements.assert_called_once()
        assert len(result) == 1
        assert result[0].id == 1  # Renumbered

    @patch("agent.desktop.accessibility.get_platform")
    @patch("agent.desktop.accessibility._get_elements_windows")
    async def test_filters_and_numbers(
        self, mock_get_elements: AsyncMock, mock_platform: MagicMock,
        mock_platform_windows: PlatformInfo,
    ) -> None:
        mock_platform.return_value = mock_platform_windows
        mock_get_elements.return_value = [
            UIElement(id=0, role="button", name="A", x=200, y=100, width=80, height=30),
            UIElement(id=0, role="button", name="B", x=100, y=100, width=80, height=30),
            UIElement(id=0, role="panel", name="", x=0, y=0, width=500, height=500),  # filtered
        ]

        result = await get_ui_elements()
        assert len(result) == 2
        # B should be first (x=100 < x=200, same y)
        assert result[0].name == "B"
        assert result[0].id == 1
        assert result[1].name == "A"
        assert result[1].id == 2

    @patch("agent.desktop.accessibility.get_platform")
    async def test_missing_uiautomation_package(
        self, mock_platform: MagicMock, mock_platform_windows: PlatformInfo,
    ) -> None:
        mock_platform.return_value = mock_platform_windows
        # Simulate ImportError from _get_elements_windows_sync
        with patch(
            "agent.desktop.accessibility._get_elements_windows",
            new_callable=AsyncMock,
            side_effect=ImportError("No module named 'uiautomation'"),
        ):
            result = await get_ui_elements()
            assert result == []


# ---------------------------------------------------------------------------
# TestGetUIElementsMacOS
# ---------------------------------------------------------------------------

class TestGetUIElementsMacOS:
    """macOS accessibility extraction via osascript."""

    @pytest.fixture
    def mock_platform_macos(self) -> PlatformInfo:
        from agent.desktop.platform_utils import OSType, PlatformInfo
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
            has_uiautomation=False,
            has_pyatspi=False,
        )

    @patch("agent.desktop.accessibility.get_platform")
    @patch("agent.desktop.accessibility._get_elements_macos")
    async def test_calls_macos_backend(
        self, mock_get_elements: AsyncMock, mock_platform: MagicMock,
        mock_platform_macos: PlatformInfo,
    ) -> None:
        mock_platform.return_value = mock_platform_macos
        mock_get_elements.return_value = [
            UIElement(id=0, role="button", name="OK", x=100, y=200, width=80, height=30),
        ]

        result = await get_ui_elements()
        mock_get_elements.assert_called_once()
        assert len(result) == 1

    @patch("agent.desktop.accessibility.get_platform")
    @patch("agent.desktop.accessibility.asyncio")
    async def test_timeout_returns_empty(
        self, mock_asyncio: MagicMock, mock_platform: MagicMock,
        mock_platform_macos: PlatformInfo,
    ) -> None:
        mock_platform.return_value = mock_platform_macos
        with patch(
            "agent.desktop.accessibility._get_elements_macos",
            new_callable=AsyncMock,
            side_effect=TimeoutError(),
        ):
            result = await get_ui_elements()
            assert result == []


# ---------------------------------------------------------------------------
# TestGetUIElementsLinux
# ---------------------------------------------------------------------------

class TestGetUIElementsLinux:
    """Linux accessibility extraction."""

    @pytest.fixture
    def mock_platform_linux(self) -> PlatformInfo:
        from agent.desktop.platform_utils import OSType, PlatformInfo
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
            has_uiautomation=False,
            has_pyatspi=True,
        )

    @patch("agent.desktop.accessibility.get_platform")
    @patch("agent.desktop.accessibility._get_elements_linux")
    async def test_calls_linux_backend(
        self, mock_get_elements: AsyncMock, mock_platform: MagicMock,
        mock_platform_linux: PlatformInfo,
    ) -> None:
        mock_platform.return_value = mock_platform_linux
        mock_get_elements.return_value = [
            UIElement(id=0, role="button", name="OK", x=100, y=200, width=80, height=30),
        ]

        result = await get_ui_elements()
        mock_get_elements.assert_called_once()
        assert len(result) == 1

    @patch("agent.desktop.accessibility.get_platform")
    @patch("agent.desktop.accessibility._get_elements_linux")
    async def test_no_api_returns_empty(
        self, mock_get_elements: AsyncMock, mock_platform: MagicMock,
        mock_platform_linux: PlatformInfo,
    ) -> None:
        mock_platform.return_value = mock_platform_linux
        mock_get_elements.return_value = []

        result = await get_ui_elements()
        assert result == []


# ---------------------------------------------------------------------------
# TestGetUIElementsGracefulFallback
# ---------------------------------------------------------------------------

class TestGetUIElementsGracefulFallback:
    """Graceful degradation for unsupported platforms."""

    @patch("agent.desktop.accessibility.get_platform")
    async def test_unknown_os_returns_empty(self, mock_platform: MagicMock) -> None:
        from agent.desktop.platform_utils import OSType, PlatformInfo
        mock_platform.return_value = PlatformInfo(
            os_type=OSType.UNKNOWN,
            has_display=True,
            display_server="",
            has_pyautogui=False,
            has_wmctrl=False,
            has_xdotool=False,
            has_osascript=False,
            screen_width=1920,
            screen_height=1080,
            scale_factor=1.0,
            has_uiautomation=False,
            has_pyatspi=False,
        )
        result = await get_ui_elements()
        assert result == []

    @patch("agent.desktop.accessibility.get_platform")
    async def test_no_display_returns_empty(self, mock_platform: MagicMock) -> None:
        from agent.desktop.platform_utils import OSType, PlatformInfo
        mock_platform.return_value = PlatformInfo(
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
            has_uiautomation=False,
            has_pyatspi=False,
        )
        result = await get_ui_elements()
        assert result == []

    @patch("agent.desktop.accessibility.get_platform")
    @patch("agent.desktop.accessibility._get_elements_windows")
    async def test_exception_returns_empty(
        self, mock_get_elements: AsyncMock, mock_platform: MagicMock,
    ) -> None:
        from agent.desktop.platform_utils import OSType, PlatformInfo
        mock_platform.return_value = PlatformInfo(
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
            has_uiautomation=True,
            has_pyatspi=False,
        )
        mock_get_elements.side_effect = RuntimeError("COM error")

        result = await get_ui_elements()
        assert result == []


# We need this import available for type hints in fixtures
from agent.desktop.platform_utils import PlatformInfo  # noqa: E402
