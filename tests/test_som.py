"""Tests for Set of Marks (SoM) screenshot annotation."""

from __future__ import annotations

import base64
import io

from PIL import Image

from agent.desktop.accessibility import UIElement
from agent.desktop.screen import Screenshot
from agent.desktop.som import (
    AnnotatedScreenshot,
    _draw_annotations,
    _get_font,
    annotate_screenshot,
    get_role_color,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_screenshot(width: int = 200, height: int = 200) -> Screenshot:
    """Create a small test screenshot."""
    img = Image.new("RGB", (width, height), color=(255, 255, 255))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    png_bytes = buf.getvalue()
    b64 = base64.b64encode(png_bytes).decode()
    return Screenshot(
        image_bytes=png_bytes,
        base64=b64,
        width=width,
        height=height,
    )


def _make_element(
    id: int = 1,
    role: str = "button",
    name: str = "OK",
    x: int = 50,
    y: int = 50,
    w: int = 80,
    h: int = 30,
) -> UIElement:
    return UIElement(id=id, role=role, name=name, x=x, y=y, width=w, height=h)


# ---------------------------------------------------------------------------
# TestAnnotatedScreenshotDataclass
# ---------------------------------------------------------------------------

class TestAnnotatedScreenshotDataclass:
    """AnnotatedScreenshot creation and element_map lookup."""

    def test_create(self) -> None:
        ss = _make_screenshot()
        elements = [_make_element(id=1), _make_element(id=2, name="Cancel", x=140)]
        annotated = AnnotatedScreenshot(
            screenshot=ss,
            annotated_base64="abc",
            annotated_bytes=b"abc",
            elements=elements,
            element_map={el.id: el for el in elements},
        )
        assert annotated.screenshot is ss
        assert len(annotated.elements) == 2
        assert annotated.annotated_base64 == "abc"

    def test_element_map_lookup(self) -> None:
        elements = [_make_element(id=1, name="A"), _make_element(id=2, name="B")]
        annotated = AnnotatedScreenshot(
            screenshot=_make_screenshot(),
            annotated_base64="",
            annotated_bytes=b"",
            elements=elements,
            element_map={el.id: el for el in elements},
        )
        assert annotated.element_map[1].name == "A"
        assert annotated.element_map[2].name == "B"
        assert 3 not in annotated.element_map


# ---------------------------------------------------------------------------
# TestRoleColors
# ---------------------------------------------------------------------------

class TestRoleColors:
    """Color assignment by role."""

    def test_button_is_blue(self) -> None:
        color = get_role_color("button")
        assert color == (59, 130, 246, 255)

    def test_text_field_is_green(self) -> None:
        color = get_role_color("text_field")
        assert color == (34, 197, 94, 255)

    def test_menu_item_is_amber(self) -> None:
        color = get_role_color("menu_item")
        assert color == (245, 158, 11, 255)

    def test_link_is_red(self) -> None:
        color = get_role_color("link")
        assert color == (239, 68, 68, 255)

    def test_checkbox_is_purple(self) -> None:
        color = get_role_color("checkbox")
        assert color == (168, 85, 247, 255)

    def test_unknown_role_gets_default(self) -> None:
        color = get_role_color("some_unknown_role")
        assert color == (107, 114, 128, 255)  # Gray default

    def test_color_is_rgba_tuple(self) -> None:
        color = get_role_color("button")
        assert len(color) == 4
        assert all(0 <= c <= 255 for c in color)


# ---------------------------------------------------------------------------
# TestAnnotateScreenshot
# ---------------------------------------------------------------------------

class TestAnnotateScreenshot:
    """Integration test for annotate_screenshot."""

    async def test_returns_valid_png(self) -> None:
        ss = _make_screenshot()
        elements = [_make_element()]
        result = await annotate_screenshot(ss, elements)

        # Should be valid PNG
        img = Image.open(io.BytesIO(result.annotated_bytes))
        assert img.format == "PNG"

    async def test_annotated_differs_from_original(self) -> None:
        ss = _make_screenshot()
        elements = [_make_element()]
        result = await annotate_screenshot(ss, elements)

        # Annotated should differ from original
        assert result.annotated_bytes != ss.image_bytes

    async def test_element_map_populated(self) -> None:
        ss = _make_screenshot()
        elements = [
            _make_element(id=1, name="A"),
            _make_element(id=2, name="B", x=120),
        ]
        result = await annotate_screenshot(ss, elements)

        assert 1 in result.element_map
        assert 2 in result.element_map
        assert result.element_map[1].name == "A"
        assert result.element_map[2].name == "B"

    async def test_empty_elements(self) -> None:
        ss = _make_screenshot()
        result = await annotate_screenshot(ss, [])

        # Should still return valid image (just the original)
        img = Image.open(io.BytesIO(result.annotated_bytes))
        assert img.format == "PNG"
        assert result.element_map == {}

    async def test_base64_is_valid(self) -> None:
        ss = _make_screenshot()
        elements = [_make_element()]
        result = await annotate_screenshot(ss, elements)

        # Base64 should decode to valid PNG
        decoded = base64.b64decode(result.annotated_base64)
        img = Image.open(io.BytesIO(decoded))
        assert img.format == "PNG"

    async def test_scale_adjustment(self) -> None:
        ss = _make_screenshot(400, 400)
        elements = [_make_element(x=100, y=100, w=80, h=30)]
        result = await annotate_screenshot(ss, elements, scale=0.5)

        # Should still produce valid output
        assert len(result.annotated_bytes) > 0
        img = Image.open(io.BytesIO(result.annotated_bytes))
        assert img.width == 400
        assert img.height == 400


# ---------------------------------------------------------------------------
# TestDrawAnnotations
# ---------------------------------------------------------------------------

class TestDrawAnnotations:
    """Low-level drawing tests."""

    def test_draws_rectangles(self) -> None:
        ss = _make_screenshot(200, 200)
        elements = [_make_element(id=1, x=50, y=50, w=80, h=30)]
        result_bytes = _draw_annotations(ss.image_bytes, elements, scale=1.0)

        img = Image.open(io.BytesIO(result_bytes))
        assert img.size == (200, 200)
        # The annotated image should differ from a plain white image
        assert result_bytes != ss.image_bytes

    def test_handles_many_elements(self) -> None:
        ss = _make_screenshot(500, 500)
        elements = [
            _make_element(id=i, x=(i % 10) * 45, y=(i // 10) * 45, w=40, h=20)
            for i in range(1, 101)
        ]
        result_bytes = _draw_annotations(ss.image_bytes, elements, scale=1.0)

        img = Image.open(io.BytesIO(result_bytes))
        assert img.size == (500, 500)

    def test_skips_zero_size_elements(self) -> None:
        ss = _make_screenshot()
        elements = [_make_element(id=1, w=0, h=0)]
        # Should not crash
        result_bytes = _draw_annotations(ss.image_bytes, elements, scale=1.0)
        assert len(result_bytes) > 0

    def test_labels_within_image_bounds(self) -> None:
        """Labels for elements at y=0 should not go above image."""
        ss = _make_screenshot()
        elements = [_make_element(id=1, x=10, y=0, w=80, h=30)]
        result_bytes = _draw_annotations(ss.image_bytes, elements, scale=1.0)

        img = Image.open(io.BytesIO(result_bytes))
        assert img.size == (200, 200)


# ---------------------------------------------------------------------------
# TestGetFont
# ---------------------------------------------------------------------------

class TestGetFont:
    """Font loading with fallback."""

    def test_returns_font(self) -> None:
        font = _get_font()
        assert font is not None

    def test_font_has_getbbox(self) -> None:
        font = _get_font()
        # All Pillow fonts should support getbbox
        bbox = font.getbbox("123")
        assert len(bbox) == 4
