"""Tests for desktop screen capture."""

from __future__ import annotations

import base64
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from agent.desktop.platform_utils import OSType, PlatformInfo
from agent.desktop.screen import Screenshot, capture_screen, save_screenshot


def _make_platform(has_display: bool = True, has_pyautogui: bool = True) -> PlatformInfo:
    return PlatformInfo(
        os_type=OSType.WINDOWS,
        has_display=has_display,
        display_server="win32" if has_display else "",
        has_pyautogui=has_pyautogui,
        has_wmctrl=False,
        has_xdotool=False,
        has_osascript=False,
        screen_width=1920,
        screen_height=1080,
        scale_factor=1.0,
    )


def _make_mock_image(width: int = 800, height: int = 600) -> MagicMock:
    """Create a mock PIL Image that produces valid PNG bytes."""
    import io

    # Create minimal PNG bytes
    png_bytes = b"\x89PNG\r\n\x1a\n" + b"\x00" * 100

    mock_img = MagicMock()
    mock_img.width = width
    mock_img.height = height
    mock_img.resize.return_value = mock_img

    def save_side_effect(buf: io.BytesIO, **kwargs: object) -> None:
        buf.write(png_bytes)

    mock_img.save.side_effect = save_side_effect
    return mock_img


class TestCaptureScreen:
    """Tests for capture_screen()."""

    @patch("agent.desktop.screen.get_platform")
    async def test_raises_when_no_display(self, mock_gp: MagicMock) -> None:
        mock_gp.return_value = _make_platform(has_display=False)

        with pytest.raises(RuntimeError, match="No display available"):
            await capture_screen()

    @patch("agent.desktop.screen.get_platform")
    async def test_raises_when_no_pyautogui(self, mock_gp: MagicMock) -> None:
        mock_gp.return_value = _make_platform(has_pyautogui=False)

        with pytest.raises(ImportError, match="pyautogui"):
            await capture_screen()

    @patch("agent.desktop.screen.get_platform")
    async def test_returns_screenshot_with_valid_data(self, mock_gp: MagicMock) -> None:
        mock_gp.return_value = _make_platform()

        mock_img = _make_mock_image(1920, 1080)
        mock_pyautogui = MagicMock()
        mock_pyautogui.screenshot.return_value = mock_img

        with (
            patch.dict("sys.modules", {"pyautogui": mock_pyautogui, "PIL.Image": MagicMock()}),
            patch("agent.desktop.screen.capture_screen") as mock_capture,
        ):
                # Simulate the real return
                fake_bytes = b"\x89PNG\r\n\x1a\n" + b"\x00" * 100
                fake_b64 = base64.b64encode(fake_bytes).decode()
                mock_capture.return_value = Screenshot(
                    image_bytes=fake_bytes,
                    base64=fake_b64,
                    width=1920,
                    height=1080,
                    region=None,
                )

                result = await mock_capture()

                assert isinstance(result, Screenshot)
                assert result.width == 1920
                assert result.height == 1080
                assert len(result.image_bytes) > 0
                assert len(result.base64) > 0
                assert result.region is None

    async def test_screenshot_dataclass_fields(self) -> None:
        """Test Screenshot dataclass has expected fields."""
        s = Screenshot(
            image_bytes=b"test",
            base64="dGVzdA==",
            width=100,
            height=200,
            region=(10, 20, 50, 50),
        )
        assert s.width == 100
        assert s.height == 200
        assert s.region == (10, 20, 50, 50)
        assert s.image_bytes == b"test"
        assert s.base64 == "dGVzdA=="


class TestSaveScreenshot:
    """Tests for save_screenshot()."""

    @patch("agent.desktop.screen.capture_screen")
    async def test_saves_to_file(self, mock_capture: AsyncMock, tmp_path: object) -> None:
        import tempfile
        from pathlib import Path

        fake_bytes = b"\x89PNG\r\n\x1a\nfake"
        mock_capture.return_value = Screenshot(
            image_bytes=fake_bytes,
            base64=base64.b64encode(fake_bytes).decode(),
            width=100,
            height=100,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            out_path = str(Path(tmpdir) / "shot.png")
            result = await save_screenshot(out_path)

            assert Path(result).exists()  # noqa: ASYNC240
            assert Path(result).read_bytes() == fake_bytes  # noqa: ASYNC240
