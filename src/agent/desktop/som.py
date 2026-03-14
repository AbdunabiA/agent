"""Set of Marks (SoM) annotation for screenshots.

Overlays numbered bounding boxes on screenshots so the LLM can pick
a UI element by number instead of guessing pixel coordinates.

Uses Pillow (already a desktop dependency) for all drawing operations.
"""

from __future__ import annotations

import asyncio
import base64
import io
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import structlog

from agent.desktop.accessibility import UIElement
from agent.desktop.screen import Screenshot

if TYPE_CHECKING:
    from PIL import ImageFont

logger = structlog.get_logger(__name__)

# Role → RGBA color mapping for bounding boxes
_ROLE_COLORS: dict[str, tuple[int, int, int, int]] = {
    "button": (59, 130, 246, 255),       # Blue
    "toolbar_button": (59, 130, 246, 255),
    "text_field": (34, 197, 94, 255),    # Green
    "menu_item": (245, 158, 11, 255),    # Amber
    "menu": (245, 158, 11, 255),
    "link": (239, 68, 68, 255),          # Red
    "checkbox": (168, 85, 247, 255),     # Purple
    "radio_button": (168, 85, 247, 255),
    "toggle": (168, 85, 247, 255),
    "dropdown": (6, 182, 212, 255),      # Cyan
    "tab": (249, 115, 22, 255),          # Orange
    "list_item": (107, 114, 128, 255),   # Gray
    "tree_item": (107, 114, 128, 255),
    "slider": (236, 72, 153, 255),       # Pink
    "spinner": (236, 72, 153, 255),
}

_DEFAULT_COLOR: tuple[int, int, int, int] = (107, 114, 128, 255)  # Gray

# Semi-transparent fill alpha
_FILL_ALPHA = 40

# Label font size
_LABEL_FONT_SIZE = 12

# Border width
_BORDER_WIDTH = 2


@dataclass
class AnnotatedScreenshot:
    """Screenshot annotated with SoM bounding boxes."""

    screenshot: Screenshot  # Original screenshot
    annotated_base64: str  # Base64-encoded annotated PNG
    annotated_bytes: bytes  # Raw annotated PNG bytes
    elements: list[UIElement]  # Elements drawn on the screenshot
    element_map: dict[int, UIElement] = field(default_factory=dict)  # ID → element


def get_role_color(role: str) -> tuple[int, int, int, int]:
    """Get the RGBA color for a given role.

    Args:
        role: Canonical role name.

    Returns:
        RGBA tuple (0-255).
    """
    return _ROLE_COLORS.get(role, _DEFAULT_COLOR)


async def annotate_screenshot(
    screenshot: Screenshot,
    elements: list[UIElement],
    scale: float = 1.0,
) -> AnnotatedScreenshot:
    """Annotate a screenshot with numbered bounding boxes.

    Args:
        screenshot: The original screenshot to annotate.
        elements: UI elements to draw (must have bounding boxes).
        scale: Coordinate scale factor (e.g., if screenshot was captured
            at 0.75x, pass 0.75 so coordinates are adjusted).

    Returns:
        AnnotatedScreenshot with the marked-up image and element map.
    """
    loop = asyncio.get_event_loop()
    annotated_bytes = await loop.run_in_executor(
        None, _draw_annotations, screenshot.image_bytes, elements, scale,
    )

    b64 = base64.b64encode(annotated_bytes).decode()

    element_map = {el.id: el for el in elements}

    logger.info(
        "screenshot_annotated",
        element_count=len(elements),
        size_kb=len(annotated_bytes) // 1024,
    )

    return AnnotatedScreenshot(
        screenshot=screenshot,
        annotated_base64=b64,
        annotated_bytes=annotated_bytes,
        elements=elements,
        element_map=element_map,
    )


def _draw_annotations(
    image_bytes: bytes,
    elements: list[UIElement],
    scale: float,
) -> bytes:
    """Draw bounding boxes and labels on the screenshot image.

    This is a blocking function, meant to run in an executor.

    Args:
        image_bytes: Original PNG bytes.
        elements: UI elements to annotate.
        scale: Coordinate scale factor.

    Returns:
        Annotated PNG bytes.
    """
    from PIL import Image, ImageDraw

    img = Image.open(io.BytesIO(image_bytes)).convert("RGBA")

    # Create overlay for semi-transparent fills
    overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
    overlay_draw = ImageDraw.Draw(overlay)

    # Draw on original for borders and labels
    draw = ImageDraw.Draw(img)

    font = _get_font()

    for el in elements:
        # Scale coordinates to match screenshot resolution
        x = int(el.x * scale)
        y = int(el.y * scale)
        w = int(el.width * scale)
        h = int(el.height * scale)

        if w <= 0 or h <= 0:
            continue

        color = get_role_color(el.role)
        r, g, b, _a = color
        fill_color = (r, g, b, _FILL_ALPHA)
        border_color = (r, g, b, 255)

        # Semi-transparent fill on overlay
        overlay_draw.rectangle(
            [x, y, x + w, y + h],
            fill=fill_color,
        )

        # Solid border
        draw.rectangle(
            [x, y, x + w, y + h],
            outline=border_color,
            width=_BORDER_WIDTH,
        )

        # Number label with pill background
        label = str(el.id)
        bbox = font.getbbox(label)
        text_w = bbox[2] - bbox[0]
        text_h = bbox[3] - bbox[1]

        pill_padding = 3
        pill_x = x
        pill_y = y - text_h - pill_padding * 2
        # Keep label within image bounds
        if pill_y < 0:
            pill_y = y

        pill_w = text_w + pill_padding * 2
        pill_h = text_h + pill_padding * 2

        # Pill background
        draw.rectangle(
            [pill_x, pill_y, pill_x + pill_w, pill_y + pill_h],
            fill=border_color,
        )

        # White text
        draw.text(
            (pill_x + pill_padding, pill_y + pill_padding),
            label,
            fill=(255, 255, 255, 255),
            font=font,
        )

    # Composite overlay onto image
    img = Image.alpha_composite(img, overlay)

    # Convert back to PNG bytes
    buf = io.BytesIO()
    img.save(buf, format="PNG", optimize=True)
    return buf.getvalue()


def _get_font() -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    """Load a suitable font for labels, with fallback.

    Tries system fonts, falls back to Pillow's built-in default.

    Returns:
        A Pillow font object.
    """
    from PIL import ImageFont

    # Platform-specific font candidates
    font_candidates = [
        # Windows
        "segoeui.ttf",
        "arial.ttf",
        "calibri.ttf",
        # macOS
        "Helvetica.ttc",
        "Helvetica Neue.ttc",
        # Linux
        "DejaVuSans.ttf",
        "LiberationSans-Regular.ttf",
        "FreeSans.ttf",
    ]

    for font_name in font_candidates:
        try:
            return ImageFont.truetype(font_name, _LABEL_FONT_SIZE)
        except OSError:
            continue

    # Fallback to default
    try:
        return ImageFont.load_default(size=_LABEL_FONT_SIZE)
    except TypeError:
        # Older Pillow doesn't support size param
        return ImageFont.load_default()
