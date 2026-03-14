"""Tests for UI element cache, click_element, ui_elements, and new accessibility tools."""

from __future__ import annotations

import time
from unittest.mock import AsyncMock, patch

from agent.desktop.accessibility import UIElement
from agent.tools.builtins.desktop import (
    _CACHE_TTL_SECONDS,
    _get_cached_element,
    _parse_target,
    _search_elements,
    _update_element_cache,
    click_element,
    find_element,
    interact,
    screen_read,
    ui_elements,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_element(
    id: int = 1, role: str = "button", name: str = "OK",
    x: int = 100, y: int = 200, w: int = 80, h: int = 30,
    focused: bool = False, enabled: bool = True,
    value: str = "",
    parent_name: str = "", parent_role: str = "",
    description: str = "",
) -> UIElement:
    return UIElement(
        id=id, role=role, name=name, x=x, y=y, width=w, height=h,
        is_enabled=enabled, is_focused=focused, value=value,
        parent_name=parent_name, parent_role=parent_role,
        description=description,
    )


def _reset_cache() -> None:
    """Reset the global element cache."""
    import agent.tools.builtins.desktop as desktop_mod
    desktop_mod._element_cache = {}
    desktop_mod._element_cache_list = []
    desktop_mod._element_cache_timestamp = 0.0


# ---------------------------------------------------------------------------
# TestElementCache
# ---------------------------------------------------------------------------

class TestElementCache:
    """Element cache store/retrieve/expiry."""

    def setup_method(self) -> None:
        _reset_cache()

    def teardown_method(self) -> None:
        _reset_cache()

    def test_store_and_retrieve(self) -> None:
        elements = [_make_element(id=1), _make_element(id=2, name="Cancel")]
        _update_element_cache(elements)

        el = _get_cached_element(1)
        assert el is not None
        assert el.name == "OK"

        el2 = _get_cached_element(2)
        assert el2 is not None
        assert el2.name == "Cancel"

    def test_invalid_id_returns_none(self) -> None:
        elements = [_make_element(id=1)]
        _update_element_cache(elements)

        el = _get_cached_element(99)
        assert el is None

    def test_empty_cache_returns_none(self) -> None:
        el = _get_cached_element(1)
        assert el is None

    def test_ttl_expiry(self) -> None:
        import agent.tools.builtins.desktop as desktop_mod

        elements = [_make_element(id=1)]
        _update_element_cache(elements)

        # Force cache to appear expired
        desktop_mod._element_cache_timestamp = time.monotonic() - _CACHE_TTL_SECONDS - 1

        el = _get_cached_element(1)
        assert el is None

    def test_replacement(self) -> None:
        elements1 = [_make_element(id=1, name="Old")]
        _update_element_cache(elements1)
        assert _get_cached_element(1).name == "Old"  # type: ignore[union-attr]

        elements2 = [_make_element(id=1, name="New"), _make_element(id=2, name="Added")]
        _update_element_cache(elements2)
        assert _get_cached_element(1).name == "New"  # type: ignore[union-attr]
        assert _get_cached_element(2).name == "Added"  # type: ignore[union-attr]


# ---------------------------------------------------------------------------
# TestUIElementsTool
# ---------------------------------------------------------------------------

class TestUIElementsTool:
    """The ui_elements tool."""

    def setup_method(self) -> None:
        _reset_cache()

    def teardown_method(self) -> None:
        _reset_cache()

    @patch("agent.desktop.accessibility.get_ui_elements", new_callable=AsyncMock)
    async def test_output_formatting(self, mock_get: AsyncMock) -> None:
        mock_get.return_value = [
            _make_element(id=1, role="button", name="OK"),
            _make_element(id=2, role="text_field", name="Search", value="hello"),
        ]

        result = await ui_elements()

        assert "Interactive UI Elements (2):" in result
        assert "[1] button:" in result
        assert '"OK"' in result
        assert "[2] text_field:" in result
        assert '"Search"' in result
        assert '= "hello"' in result
        assert "click_element" in result

    @patch("agent.desktop.accessibility.get_ui_elements", new_callable=AsyncMock)
    async def test_empty_list(self, mock_get: AsyncMock) -> None:
        mock_get.return_value = []

        result = await ui_elements()

        assert "No UI elements detected" in result

    @patch("agent.desktop.accessibility.get_ui_elements", new_callable=AsyncMock)
    async def test_focused_element_display(self, mock_get: AsyncMock) -> None:
        mock_get.return_value = [
            _make_element(id=1, focused=True),
        ]

        result = await ui_elements()
        assert "[FOCUSED]" in result

    @patch("agent.desktop.accessibility.get_ui_elements", new_callable=AsyncMock)
    async def test_disabled_element_display(self, mock_get: AsyncMock) -> None:
        mock_get.return_value = [
            _make_element(id=1, enabled=False),
        ]

        result = await ui_elements()
        assert "[DISABLED]" in result

    @patch("agent.desktop.accessibility.get_ui_elements", new_callable=AsyncMock)
    async def test_updates_cache(self, mock_get: AsyncMock) -> None:
        elements = [_make_element(id=1), _make_element(id=2, name="Cancel")]
        mock_get.return_value = elements

        await ui_elements()

        # Cache should be populated
        assert _get_cached_element(1) is not None
        assert _get_cached_element(2) is not None


# ---------------------------------------------------------------------------
# TestClickElementTool
# ---------------------------------------------------------------------------

class TestClickElementTool:
    """The click_element tool."""

    def setup_method(self) -> None:
        _reset_cache()

    def teardown_method(self) -> None:
        _reset_cache()

    @patch("agent.desktop.mouse.click", new_callable=AsyncMock)
    async def test_center_calculation(self, mock_click: AsyncMock) -> None:
        mock_click.return_value = "Clicked at (140, 215)"
        _update_element_cache([_make_element(id=1, x=100, y=200, w=80, h=30)])

        result = await click_element(element_id=1)

        # Center of (100, 200, 80, 30) = (140, 215)
        mock_click.assert_called_once_with(x=140, y=215, button="left", clicks=1)
        assert "Clicked [1]" in result
        assert '"OK"' in result

    async def test_cache_empty_error(self) -> None:
        result = await click_element(element_id=1)
        assert "Error" in result
        assert "No element cache" in result

    async def test_expired_cache(self) -> None:
        import agent.tools.builtins.desktop as desktop_mod

        _update_element_cache([_make_element(id=1)])
        desktop_mod._element_cache_timestamp = time.monotonic() - _CACHE_TTL_SECONDS - 1

        result = await click_element(element_id=1)
        assert "Error" in result
        assert "expired" in result

    async def test_invalid_id(self) -> None:
        _update_element_cache([_make_element(id=1), _make_element(id=2, name="B")])

        result = await click_element(element_id=99)
        assert "Error" in result
        assert "not found" in result
        assert "1-2" in result  # valid range

    @patch("agent.desktop.mouse.click", new_callable=AsyncMock)
    async def test_double_click(self, mock_click: AsyncMock) -> None:
        mock_click.return_value = "Double-clicked"
        _update_element_cache([_make_element(id=1)])

        await click_element(element_id=1, clicks=2)
        mock_click.assert_called_once_with(x=140, y=215, button="left", clicks=2)

    @patch("agent.desktop.mouse.click", new_callable=AsyncMock)
    async def test_right_click(self, mock_click: AsyncMock) -> None:
        mock_click.return_value = "Right-clicked"
        _update_element_cache([_make_element(id=1)])

        await click_element(element_id=1, button="right")
        mock_click.assert_called_once_with(x=140, y=215, button="right", clicks=1)


# ---------------------------------------------------------------------------
# TestScreenCaptureAnnotate
# ---------------------------------------------------------------------------

class TestScreenCaptureAnnotate:
    """screen_capture with annotate flag."""

    def setup_method(self) -> None:
        _reset_cache()

    def teardown_method(self) -> None:
        _reset_cache()

    @patch("agent.desktop.som.annotate_screenshot", new_callable=AsyncMock)
    @patch("agent.desktop.accessibility.get_ui_elements", new_callable=AsyncMock)
    @patch("agent.desktop.screen.capture_screen", new_callable=AsyncMock)
    async def test_annotate_true_returns_multimodal(
        self,
        mock_capture: AsyncMock,
        mock_get_elements: AsyncMock,
        mock_annotate: AsyncMock,
    ) -> None:
        from agent.desktop.screen import Screenshot
        from agent.desktop.som import AnnotatedScreenshot
        from agent.tools.executor import MultimodalToolOutput

        ss = Screenshot(
            image_bytes=b"png_data",
            base64="base64_data",
            width=200,
            height=200,
        )
        mock_capture.return_value = ss

        elements = [_make_element(id=1)]
        mock_get_elements.return_value = elements

        annotated = AnnotatedScreenshot(
            screenshot=ss,
            annotated_base64="annotated_b64",
            annotated_bytes=b"annotated_png",
            elements=elements,
            element_map={1: elements[0]},
        )
        mock_annotate.return_value = annotated

        # Import the actual function to test
        from agent.tools.builtins.desktop import screen_capture as sc_tool

        result = await sc_tool(annotate=True)

        assert isinstance(result, MultimodalToolOutput)
        assert "Annotated screenshot" in result.text
        assert "1 UI elements" in result.text
        assert len(result.images) == 1
        assert result.images[0].base64_data == "annotated_b64"

    @patch("agent.desktop.screen.capture_screen", new_callable=AsyncMock)
    async def test_annotate_false_unchanged(self, mock_capture: AsyncMock) -> None:
        from agent.desktop.screen import Screenshot
        from agent.tools.executor import MultimodalToolOutput

        ss = Screenshot(
            image_bytes=b"png_data",
            base64="base64_data",
            width=200,
            height=200,
        )
        mock_capture.return_value = ss

        from agent.tools.builtins.desktop import screen_capture as sc_tool

        result = await sc_tool(annotate=False)

        assert isinstance(result, MultimodalToolOutput)
        assert "Screenshot captured" in result.text
        assert result.images[0].base64_data == "base64_data"

    @patch("agent.desktop.accessibility.get_ui_elements", new_callable=AsyncMock)
    @patch("agent.desktop.screen.capture_screen", new_callable=AsyncMock)
    async def test_annotate_no_elements_falls_back(
        self, mock_capture: AsyncMock, mock_get_elements: AsyncMock,
    ) -> None:
        from agent.desktop.screen import Screenshot
        from agent.tools.executor import MultimodalToolOutput

        ss = Screenshot(
            image_bytes=b"png_data",
            base64="base64_data",
            width=200,
            height=200,
        )
        mock_capture.return_value = ss
        mock_get_elements.return_value = []  # No elements found

        from agent.tools.builtins.desktop import screen_capture as sc_tool

        result = await sc_tool(annotate=True)

        # Falls through to normal screenshot when no elements
        assert isinstance(result, MultimodalToolOutput)
        assert "Screenshot captured" in result.text


# ---------------------------------------------------------------------------
# TestElementSearch
# ---------------------------------------------------------------------------

class TestElementSearch:
    """_search_elements cache search."""

    def setup_method(self) -> None:
        _reset_cache()

    def teardown_method(self) -> None:
        _reset_cache()

    def test_search_by_name(self) -> None:
        _update_element_cache([
            _make_element(id=1, name="Save"),
            _make_element(id=2, name="Cancel"),
            _make_element(id=3, name="Save As"),
        ])
        results = _search_elements(name="Save")
        assert len(results) == 2
        assert results[0].name == "Save"
        assert results[1].name == "Save As"

    def test_search_by_name_case_insensitive(self) -> None:
        _update_element_cache([
            _make_element(id=1, name="OK"),
        ])
        results = _search_elements(name="ok")
        assert len(results) == 1

    def test_search_by_role(self) -> None:
        _update_element_cache([
            _make_element(id=1, role="button", name="OK"),
            _make_element(id=2, role="text_field", name="Search"),
            _make_element(id=3, role="button", name="Cancel"),
        ])
        results = _search_elements(role="button")
        assert len(results) == 2
        assert all(r.role == "button" for r in results)

    def test_search_combined(self) -> None:
        _update_element_cache([
            _make_element(id=1, role="button", name="OK"),
            _make_element(id=2, role="button", name="Cancel"),
            _make_element(id=3, role="text_field", name="OK Field"),
        ])
        results = _search_elements(name="OK", role="button")
        assert len(results) == 1
        assert results[0].name == "OK"
        assert results[0].role == "button"

    def test_search_no_match(self) -> None:
        _update_element_cache([
            _make_element(id=1, name="OK"),
        ])
        results = _search_elements(name="NonExistent")
        assert len(results) == 0

    def test_search_limit(self) -> None:
        elements = [_make_element(id=i, name=f"Btn{i}") for i in range(1, 20)]
        _update_element_cache(elements)
        results = _search_elements(name="Btn", limit=5)
        assert len(results) == 5


# ---------------------------------------------------------------------------
# TestTargetParsing
# ---------------------------------------------------------------------------

class TestTargetParsing:
    """_parse_target natural-language parsing."""

    def test_save_button(self) -> None:
        name, role = _parse_target("Save button")
        assert name == "Save"
        assert role == "button"

    def test_name_field(self) -> None:
        name, role = _parse_target("Name field")
        assert name == "Name"
        assert role == "text_field"

    def test_file_menu(self) -> None:
        name, role = _parse_target("File menu")
        assert name == "File"
        assert role == "menu_item"

    def test_plain_name(self) -> None:
        name, role = _parse_target("OK")
        assert name == "OK"
        assert role is None

    def test_two_word_role(self) -> None:
        name, role = _parse_target("Username text field")
        assert name == "Username"
        assert role == "text_field"

    def test_btn_alias(self) -> None:
        name, role = _parse_target("Submit btn")
        assert name == "Submit"
        assert role == "button"

    def test_checkbox(self) -> None:
        name, role = _parse_target("Remember me checkbox")
        assert name == "Remember me"
        assert role == "checkbox"

    def test_dropdown_alias(self) -> None:
        name, role = _parse_target("Country select")
        assert name == "Country"
        assert role == "dropdown"

    def test_toggle_alias(self) -> None:
        name, role = _parse_target("Dark mode switch")
        assert name == "Dark mode"
        assert role == "toggle"

    def test_empty_string(self) -> None:
        name, role = _parse_target("")
        assert name is None
        assert role is None


# ---------------------------------------------------------------------------
# TestFindElementTool
# ---------------------------------------------------------------------------

class TestFindElementTool:
    """find_element tool."""

    def setup_method(self) -> None:
        _reset_cache()

    def teardown_method(self) -> None:
        _reset_cache()

    async def test_no_params_error(self) -> None:
        result = await find_element()
        assert "Error" in result

    @patch("agent.desktop.accessibility.get_ui_elements", new_callable=AsyncMock)
    async def test_search_by_name(self, mock_get: AsyncMock) -> None:
        elements = [
            _make_element(id=1, name="Save", parent_name="Toolbar"),
            _make_element(id=2, name="Cancel"),
        ]
        mock_get.return_value = elements

        result = await find_element(name="Save")
        assert "1 matching" in result
        assert '"Save"' in result
        assert '(in "Toolbar")' in result

    @patch("agent.desktop.accessibility.get_ui_elements", new_callable=AsyncMock)
    async def test_search_by_role(self, mock_get: AsyncMock) -> None:
        elements = [
            _make_element(id=1, role="button", name="OK"),
            _make_element(id=2, role="text_field", name="Search"),
        ]
        mock_get.return_value = elements

        result = await find_element(role="button")
        assert "1 matching" in result
        assert "button" in result

    @patch("agent.desktop.accessibility.get_ui_elements", new_callable=AsyncMock)
    async def test_auto_refresh(self, mock_get: AsyncMock) -> None:
        """Cache should be refreshed if empty."""
        elements = [_make_element(id=1, name="OK")]
        mock_get.return_value = elements

        result = await find_element(name="OK")
        mock_get.assert_called_once()
        assert '"OK"' in result

    @patch("agent.desktop.accessibility.get_ui_elements", new_callable=AsyncMock)
    async def test_no_match(self, mock_get: AsyncMock) -> None:
        mock_get.return_value = [_make_element(id=1, name="OK")]

        result = await find_element(name="NonExistent")
        assert "No elements found" in result
        assert "screen_capture" in result

    @patch("agent.desktop.accessibility.get_ui_elements", new_callable=AsyncMock)
    async def test_uses_existing_cache(self, mock_get: AsyncMock) -> None:
        """Should not call get_ui_elements if cache is valid."""
        _update_element_cache([_make_element(id=1, name="OK")])

        result = await find_element(name="OK")
        mock_get.assert_not_called()
        assert '"OK"' in result


# ---------------------------------------------------------------------------
# TestInteractTool
# ---------------------------------------------------------------------------

class TestInteractTool:
    """interact composite tool."""

    def setup_method(self) -> None:
        _reset_cache()

    def teardown_method(self) -> None:
        _reset_cache()

    @patch("agent.desktop.mouse.click", new_callable=AsyncMock)
    @patch("agent.desktop.accessibility.get_ui_elements", new_callable=AsyncMock)
    async def test_click_action(
        self, mock_get: AsyncMock, mock_click: AsyncMock,
    ) -> None:
        mock_get.return_value = [_make_element(id=1, name="OK")]
        mock_click.return_value = "Clicked"

        result = await interact(target="OK", action="click")
        assert "Clicked" in result
        assert '"OK"' in result
        mock_click.assert_called_once()

    @patch("agent.desktop.keyboard.type_text", new_callable=AsyncMock)
    @patch("agent.desktop.mouse.click", new_callable=AsyncMock)
    @patch("agent.desktop.accessibility.get_ui_elements", new_callable=AsyncMock)
    async def test_type_action(
        self, mock_get: AsyncMock, mock_click: AsyncMock, mock_type: AsyncMock,
    ) -> None:
        mock_get.return_value = [
            _make_element(id=1, role="text_field", name="Search"),
        ]
        mock_click.return_value = "Clicked"
        mock_type.return_value = "Typed"

        result = await interact(target="Search field", action="type", text="hello")
        assert "Focused" in result
        assert "typed 5 characters" in result
        mock_click.assert_called_once()
        mock_type.assert_called_once_with("hello")

    async def test_type_without_text_error(self) -> None:
        result = await interact(target="Search", action="type")
        assert "Error" in result
        assert "text" in result

    @patch("agent.desktop.mouse.click", new_callable=AsyncMock)
    @patch("agent.desktop.accessibility.get_ui_elements", new_callable=AsyncMock)
    async def test_focus_action(
        self, mock_get: AsyncMock, mock_click: AsyncMock,
    ) -> None:
        mock_get.return_value = [_make_element(id=1, name="Search")]
        mock_click.return_value = "Clicked"

        result = await interact(target="Search", action="focus")
        assert "Focused" in result

    @patch("agent.desktop.accessibility.get_ui_elements", new_callable=AsyncMock)
    async def test_read_action(self, mock_get: AsyncMock) -> None:
        mock_get.return_value = [
            _make_element(
                id=1, name="Search", role="text_field", value="hello",
                parent_name="Toolbar", parent_role="toolbar_button",
                description="Search the web",
            ),
        ]

        result = await interact(target="Search", action="read")
        assert "properties" in result
        assert '"Search"' in result
        assert "text_field" in result
        assert '"hello"' in result
        assert '"Toolbar"' in result
        assert '"Search the web"' in result

    @patch("agent.desktop.accessibility.get_ui_elements", new_callable=AsyncMock)
    async def test_element_not_found(self, mock_get: AsyncMock) -> None:
        mock_get.return_value = [_make_element(id=1, name="OK")]

        result = await interact(target="NonExistent button")
        assert "No element found" in result
        assert "screen_capture" in result

    async def test_unknown_action(self) -> None:
        _update_element_cache([_make_element(id=1, name="OK")])

        result = await interact(target="OK", action="explode")
        assert "Unknown action" in result

    @patch("agent.desktop.mouse.click", new_callable=AsyncMock)
    async def test_uses_existing_cache(self, mock_click: AsyncMock) -> None:
        """Should use cache without refreshing if valid."""
        _update_element_cache([_make_element(id=1, name="OK")])
        mock_click.return_value = "Clicked"

        result = await interact(target="OK")
        assert "Clicked" in result

    @patch("agent.desktop.mouse.click", new_callable=AsyncMock)
    @patch("agent.desktop.accessibility.get_ui_elements", new_callable=AsyncMock)
    async def test_target_parsing_integration(
        self, mock_get: AsyncMock, mock_click: AsyncMock,
    ) -> None:
        """'Save button' should find element named Save with role button."""
        mock_get.return_value = [
            _make_element(id=1, role="button", name="Save"),
            _make_element(id=2, role="text_field", name="Save Path"),
        ]
        mock_click.return_value = "Clicked"

        result = await interact(target="Save button")
        assert "button" in result
        assert '"Save"' in result


# ---------------------------------------------------------------------------
# TestScreenReadTool
# ---------------------------------------------------------------------------

class TestScreenReadTool:
    """screen_read tool."""

    def setup_method(self) -> None:
        _reset_cache()

    def teardown_method(self) -> None:
        _reset_cache()

    @patch("agent.desktop.accessibility.get_ui_elements", new_callable=AsyncMock)
    @patch("agent.desktop.windows.list_windows", new_callable=AsyncMock)
    async def test_normal_output(
        self, mock_windows: AsyncMock, mock_get: AsyncMock,
    ) -> None:
        from agent.desktop.windows import WindowInfo

        mock_windows.return_value = [
            WindowInfo(
                id="1", title="Untitled - Notepad", app_name="notepad.exe",
                x=0, y=0, width=800, height=600, is_active=True,
                is_minimized=False,
            ),
        ]
        mock_get.return_value = [
            _make_element(id=1, role="button", name="OK"),
            _make_element(id=2, role="text_field", name="Search", focused=True),
            _make_element(id=3, role="button", name="Cancel"),
        ]

        result = await screen_read()

        assert "Untitled - Notepad" in result
        assert "Focused element" in result
        assert '"Search"' in result
        assert "button: 2" in result
        assert "text_field: 1" in result
        assert "Element list:" in result

    @patch("agent.desktop.accessibility.get_ui_elements", new_callable=AsyncMock)
    @patch("agent.desktop.windows.list_windows", new_callable=AsyncMock)
    async def test_empty_elements(
        self, mock_windows: AsyncMock, mock_get: AsyncMock,
    ) -> None:
        mock_windows.return_value = []
        mock_get.return_value = []

        result = await screen_read()
        assert "No UI elements detected" in result

    @patch("agent.desktop.accessibility.get_ui_elements", new_callable=AsyncMock)
    @patch("agent.desktop.windows.list_windows", new_callable=AsyncMock)
    async def test_no_focused_element(
        self, mock_windows: AsyncMock, mock_get: AsyncMock,
    ) -> None:
        mock_windows.return_value = []
        mock_get.return_value = [
            _make_element(id=1, name="OK", focused=False),
        ]

        result = await screen_read()
        assert "Focused element: none detected" in result

    @patch("agent.desktop.accessibility.get_ui_elements", new_callable=AsyncMock)
    @patch("agent.desktop.windows.list_windows", new_callable=AsyncMock)
    async def test_parent_context_in_output(
        self, mock_windows: AsyncMock, mock_get: AsyncMock,
    ) -> None:
        mock_windows.return_value = []
        mock_get.return_value = [
            _make_element(id=1, name="Save", parent_name="Toolbar"),
        ]

        result = await screen_read()
        assert '(in "Toolbar")' in result

    @patch("agent.desktop.accessibility.get_ui_elements", new_callable=AsyncMock)
    @patch("agent.desktop.windows.list_windows", new_callable=AsyncMock)
    async def test_updates_cache(
        self, mock_windows: AsyncMock, mock_get: AsyncMock,
    ) -> None:
        mock_windows.return_value = []
        elements = [_make_element(id=1)]
        mock_get.return_value = elements

        await screen_read()
        assert _get_cached_element(1) is not None
