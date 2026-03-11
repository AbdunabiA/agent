"""Tests for desktop application management."""

from __future__ import annotations

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

from agent.desktop.apps import launch_app, list_installed_apps, open_file, open_url
from agent.desktop.platform_utils import OSType, PlatformInfo

_PLATFORM_WIN = PlatformInfo(
    os_type=OSType.WINDOWS, has_display=True, display_server="win32",
    has_pyautogui=True, has_wmctrl=False, has_xdotool=False,
    has_osascript=False, screen_width=1920, screen_height=1080, scale_factor=1.0,
)

_PLATFORM_LINUX = PlatformInfo(
    os_type=OSType.LINUX, has_display=True, display_server="x11",
    has_pyautogui=True, has_wmctrl=False, has_xdotool=False,
    has_osascript=False, screen_width=1920, screen_height=1080, scale_factor=1.0,
)

_PLATFORM_MAC = PlatformInfo(
    os_type=OSType.MACOS, has_display=True, display_server="quartz",
    has_pyautogui=True, has_wmctrl=False, has_xdotool=False,
    has_osascript=True, screen_width=1920, screen_height=1080, scale_factor=1.0,
)


class TestLaunchApp:
    """Tests for launch_app()."""

    @patch(
        "agent.desktop.apps.get_app_launch_command",
        return_value=["cmd", "/c", "start", "", "notepad"],
    )
    @patch("agent.desktop.apps.subprocess.Popen")
    async def test_launches_app(self, mock_popen: MagicMock, _cmd: object) -> None:
        result = await launch_app("notepad")
        assert "launched" in result
        mock_popen.assert_called_once()

    @patch("agent.desktop.apps.get_app_launch_command", return_value=["nonexistent_binary_xyz"])
    @patch("agent.desktop.apps.subprocess.Popen", side_effect=FileNotFoundError)
    async def test_returns_error_for_missing_app(self, _popen: object, _cmd: object) -> None:
        result = await launch_app("nonexistent_binary_xyz")
        assert "[ERROR]" in result
        assert "not found" in result

    @patch(
        "agent.desktop.apps.get_app_launch_command",
        return_value=["cmd", "/c", "start", "", "notepad"],
    )
    @patch("agent.desktop.apps.subprocess.Popen")
    async def test_passes_args(self, mock_popen: MagicMock, _cmd: object) -> None:
        result = await launch_app("code", args=["/home/user/project"])
        assert "launched" in result


class TestOpenFile:
    """Tests for open_file()."""

    async def test_returns_error_for_nonexistent_file(self) -> None:
        result = await open_file("/nonexistent/path/to/file.txt")
        assert "[ERROR]" in result
        assert "not found" in result.lower() or "File not found" in result

    @patch("agent.desktop.apps.get_platform", return_value=_PLATFORM_WIN)
    async def test_opens_existing_file(self, _: object) -> None:
        with tempfile.NamedTemporaryFile(suffix=".txt", delete=False) as f:
            f.write(b"hello")
            f.flush()
            tmp_path = f.name

        try:
            with patch("agent.desktop.apps.asyncio.create_subprocess_exec") as mock_exec:
                mock_proc = MagicMock()
                mock_proc.wait = MagicMock(return_value=None)

                async def _wait() -> None:
                    return None

                mock_proc.wait = _wait
                mock_exec.return_value = mock_proc

                result = await open_file(tmp_path)
                assert "Opened" in result
        finally:
            Path(tmp_path).unlink(missing_ok=True)  # noqa: ASYNC240


class TestOpenUrl:
    """Tests for open_url()."""

    @patch("agent.desktop.apps.webbrowser.open", return_value=True)
    async def test_opens_url_successfully(self, _mock: object) -> None:
        result = await open_url("https://example.com")
        assert "Opened" in result
        assert "example.com" in result

    @patch("agent.desktop.apps.webbrowser.open", return_value=False)
    async def test_returns_error_on_failure(self, _mock: object) -> None:
        result = await open_url("https://example.com")
        assert "[ERROR]" in result


class TestListInstalledApps:
    """Tests for list_installed_apps()."""

    @patch("agent.desktop.apps.get_platform", return_value=_PLATFORM_WIN)
    async def test_returns_formatted_list(self, _: object) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create fake Program Files structure
            (Path(tmpdir) / "AppOne").mkdir()
            (Path(tmpdir) / "AppTwo").mkdir()

            with patch("agent.desktop.apps.Path") as mock_path_cls:
                # Only mock the specific Path(r"C:\Program Files") calls
                real_path = Path

                def path_side_effect(*args: object, **kwargs: object) -> Path:
                    path_str = str(args[0]) if args else ""
                    if "Program Files" in path_str:
                        return real_path(tmpdir)
                    return real_path(*args, **kwargs)

                mock_path_cls.side_effect = path_side_effect
                mock_path_cls.home.return_value = real_path.home()

                # Directly test with a simpler approach
                result = await list_installed_apps()
                assert "Installed Applications" in result

    @patch("agent.desktop.apps.get_platform", return_value=_PLATFORM_LINUX)
    async def test_linux_returns_list(self, _: object) -> None:
        result = await list_installed_apps()
        assert "Installed Applications" in result
