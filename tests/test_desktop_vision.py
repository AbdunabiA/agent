"""Tests for desktop vision LLM integration."""

from __future__ import annotations

from dataclasses import dataclass
from unittest.mock import AsyncMock, MagicMock

from agent.desktop.screen import Screenshot
from agent.desktop.vision import VisionAnalyzer


def _make_screenshot() -> Screenshot:
    return Screenshot(
        image_bytes=b"\x89PNG\r\n\x1a\nfake",
        base64="iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg==",
        width=1920,
        height=1080,
    )


@dataclass
class FakeLLMResponse:
    content: str


class TestDescribeScreen:
    """Tests for VisionAnalyzer.describe_screen()."""

    async def test_returns_description(self) -> None:
        mock_llm = MagicMock()
        mock_llm.completion = AsyncMock(return_value=FakeLLMResponse(
            content="Firefox browser showing Google homepage with search bar in center."
        ))

        analyzer = VisionAnalyzer(mock_llm)
        result = await analyzer.describe_screen(_make_screenshot())

        assert "Firefox" in result
        assert "Google" in result
        mock_llm.completion.assert_called_once()

        # Verify image was sent in the message
        call_args = mock_llm.completion.call_args
        messages = call_args[0][0]
        assert len(messages) == 1
        content = messages[0]["content"]
        assert any(item.get("type") == "image_url" for item in content)


class TestFindElement:
    """Tests for VisionAnalyzer.find_element()."""

    async def test_finds_element_with_coords(self) -> None:
        mock_llm = MagicMock()
        mock_llm.completion = AsyncMock(return_value=FakeLLMResponse(
            content='{"found": true, "x": 960, "y": 540, "width": 200, '
                    '"height": 35, "description": "Google search input field"}'
        ))

        analyzer = VisionAnalyzer(mock_llm)
        result = await analyzer.find_element(_make_screenshot(), "search bar")

        assert result is not None
        assert result["found"] is True
        assert result["x"] == 960
        assert result["y"] == 540
        assert "search" in str(result["description"]).lower()

    async def test_returns_none_when_not_found(self) -> None:
        mock_llm = MagicMock()
        mock_llm.completion = AsyncMock(return_value=FakeLLMResponse(
            content='{"found": false, "description": "No search bar visible on screen"}'
        ))

        analyzer = VisionAnalyzer(mock_llm)
        result = await analyzer.find_element(_make_screenshot(), "search bar")

        assert result is None

    async def test_handles_markdown_fenced_json(self) -> None:
        mock_llm = MagicMock()
        mock_llm.completion = AsyncMock(return_value=FakeLLMResponse(
            content='```json\n{"found": true, "x": 100, "y": 200, '
                    '"width": 50, "height": 30, "description": "button"}\n```'
        ))

        analyzer = VisionAnalyzer(mock_llm)
        result = await analyzer.find_element(_make_screenshot(), "button")

        assert result is not None
        assert result["found"] is True
        assert result["x"] == 100

    async def test_handles_invalid_json(self) -> None:
        mock_llm = MagicMock()
        mock_llm.completion = AsyncMock(return_value=FakeLLMResponse(
            content="I can see a button at approximately..."
        ))

        analyzer = VisionAnalyzer(mock_llm)
        result = await analyzer.find_element(_make_screenshot(), "button")

        assert result is None

    async def test_uses_custom_model(self) -> None:
        mock_llm = MagicMock()
        mock_llm.completion = AsyncMock(return_value=FakeLLMResponse(
            content='{"found": true, "x": 50, "y": 50, "width": 10, '
                    '"height": 10, "description": "icon"}'
        ))

        analyzer = VisionAnalyzer(mock_llm, model="gpt-4o")
        await analyzer.find_element(_make_screenshot(), "icon")

        call_kwargs = mock_llm.completion.call_args
        model_val = call_kwargs[1].get("model") or call_kwargs.kwargs.get("model")
        assert model_val == "gpt-4o"


class TestGetClickableElements:
    """Tests for VisionAnalyzer.get_clickable_elements()."""

    async def test_returns_element_list(self) -> None:
        mock_llm = MagicMock()
        mock_llm.completion = AsyncMock(return_value=FakeLLMResponse(
            content='[{"label": "Submit", "type": "button", "x": 400, "y": 300, '
                    '"width": 100, "height": 35}, '
                    '{"label": "Search", "type": "input", "x": 500, "y": 100, '
                    '"width": 200, "height": 30}]'
        ))

        analyzer = VisionAnalyzer(mock_llm)
        result = await analyzer.get_clickable_elements(_make_screenshot())

        assert len(result) == 2
        assert result[0]["label"] == "Submit"
        assert result[1]["type"] == "input"

    async def test_returns_empty_on_invalid_json(self) -> None:
        mock_llm = MagicMock()
        mock_llm.completion = AsyncMock(return_value=FakeLLMResponse(
            content="There are several buttons visible..."
        ))

        analyzer = VisionAnalyzer(mock_llm)
        result = await analyzer.get_clickable_elements(_make_screenshot())

        assert result == []

    async def test_handles_markdown_fenced_array(self) -> None:
        mock_llm = MagicMock()
        mock_llm.completion = AsyncMock(return_value=FakeLLMResponse(
            content='```json\n[{"label": "OK", "type": "button", '
                    '"x": 100, "y": 100, "width": 50, "height": 25}]\n```'
        ))

        analyzer = VisionAnalyzer(mock_llm)
        result = await analyzer.get_clickable_elements(_make_screenshot())

        assert len(result) == 1
        assert result[0]["label"] == "OK"
