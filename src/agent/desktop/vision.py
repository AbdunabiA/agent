"""Vision analysis of screenshots using multimodal LLMs.

Sends screenshots to the LLM for describing screen content,
finding UI elements by description, and getting coordinates to click.
"""

from __future__ import annotations

import json

import structlog

from agent.desktop.screen import Screenshot
from agent.llm.provider import LLMProvider

logger = structlog.get_logger(__name__)


class VisionAnalyzer:
    """Uses a vision-capable LLM to understand screenshots."""

    def __init__(self, llm: LLMProvider, model: str | None = None) -> None:
        """Initialize with an LLM provider.

        Args:
            llm: The LLM provider to use for vision analysis.
            model: Optional model override (for using a vision-specific model).
        """
        self.llm = llm
        self.model = model

    async def describe_screen(self, screenshot: Screenshot) -> str:
        """Get a description of what's currently visible on screen.

        Args:
            screenshot: Captured screenshot.

        Returns:
            Structured description of screen content.
        """
        messages = [{
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/png;base64,{screenshot.base64}",
                    },
                },
                {
                    "type": "text",
                    "text": (
                        "Describe what you see on this computer screen. Include:\n"
                        "1. What application is in the foreground\n"
                        "2. What the main content shows\n"
                        "3. Key UI elements visible (buttons, menus, input fields)\n"
                        "4. Any dialogs, popups, or notifications\n"
                        "Be specific and concise."
                    ),
                },
            ],
        }]

        response = await self.llm.completion(messages, model=self.model)
        return response.content

    async def find_element(
        self,
        screenshot: Screenshot,
        description: str,
    ) -> dict[str, object] | None:
        """Find a UI element by description and return its approximate coordinates.

        Args:
            screenshot: Current screenshot.
            description: What to find, e.g., "the search bar",
                        "the Submit button", "the close button on the dialog".

        Returns:
            Dict with "found", "x", "y", "width", "height", "description"
            or None if element not found.
        """
        messages = [{
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/png;base64,{screenshot.base64}",
                    },
                },
                {
                    "type": "text",
                    "text": (
                        f'Find this UI element on the screen: "{description}"\n\n'
                        f"The screen resolution is {screenshot.width}x{screenshot.height}.\n\n"
                        "Respond ONLY with JSON (no markdown, no explanation):\n"
                        'If found: {"found": true, "x": <center_x>, "y": <center_y>, '
                        '"width": <approx_width>, "height": <approx_height>, '
                        '"description": "<what the element is>"}\n'
                        'If not found: {"found": false, "description": "<why not found>"}'
                    ),
                },
            ],
        }]

        response = await self.llm.completion(messages, model=self.model)

        try:
            text = response.content.strip()
            # Strip markdown code fences if present
            if text.startswith("```"):
                text = text.split("\n", 1)[1] if "\n" in text else text[3:]
                if text.endswith("```"):
                    text = text[:-3]
                text = text.strip()

            result = json.loads(text)

            if result.get("found"):
                logger.info(
                    "element_found",
                    description=description,
                    x=result.get("x"),
                    y=result.get("y"),
                )
                return result
            else:
                logger.info(
                    "element_not_found",
                    description=description,
                    reason=result.get("description"),
                )
                return None
        except json.JSONDecodeError:
            logger.error("vision_json_parse_error", response=response.content[:200])
            return None

    async def get_clickable_elements(self, screenshot: Screenshot) -> list[dict[str, object]]:
        """Identify all clickable elements on screen with coordinates.

        Args:
            screenshot: Current screenshot.

        Returns:
            List of dicts with "label", "type", "x", "y", "width", "height".
        """
        messages = [{
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/png;base64,{screenshot.base64}",
                    },
                },
                {
                    "type": "text",
                    "text": (
                        f"Screen resolution: {screenshot.width}x{screenshot.height}.\n\n"
                        "List all clickable/interactive UI elements visible on screen.\n"
                        "Respond ONLY with a JSON array (no markdown):\n"
                        '[{"label": "element text", '
                        '"type": "button|link|input|menu|tab|checkbox", '
                        '"x": center_x, "y": center_y, "width": w, "height": h}]'
                    ),
                },
            ],
        }]

        response = await self.llm.completion(messages, model=self.model)

        try:
            text = response.content.strip()
            if text.startswith("```"):
                text = text.split("\n", 1)[1] if "\n" in text else text[3:]
                if text.endswith("```"):
                    text = text[:-3]
                text = text.strip()

            result = json.loads(text)
            if isinstance(result, list):
                return result
            return []
        except json.JSONDecodeError:
            logger.error("vision_parse_error", response=response.content[:200])
            return []
