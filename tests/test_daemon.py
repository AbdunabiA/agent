"""Tests for the daemon service management module."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

from agent.core.daemon import (
    DaemonStatus,
    _get_platform,
    daemon_install,
    daemon_start,
    daemon_status,
    daemon_stop,
    daemon_uninstall,
)


class TestDaemonStatus:
    """Tests for the DaemonStatus dataclass."""

    def test_defaults(self) -> None:
        status = DaemonStatus(installed=False, running=False)
        assert status.installed is False
        assert status.running is False
        assert status.pid is None
        assert status.service_path is None
        assert status.log_path is None

    def test_with_pid(self) -> None:
        status = DaemonStatus(installed=True, running=True, pid=12345)
        assert status.pid == 12345


class TestGetPlatform:
    """Tests for the _get_platform dispatcher."""

    @patch("agent.core.daemon.platform.system", return_value="Darwin")
    def test_macos_detected(self, _mock: MagicMock) -> None:
        assert _get_platform() == "macos"

    @patch("agent.core.daemon.platform.system", return_value="Linux")
    def test_linux_detected(self, _mock: MagicMock) -> None:
        assert _get_platform() == "linux"

    @patch("agent.core.daemon.platform.system", return_value="Windows")
    def test_unsupported_platform(self, _mock: MagicMock) -> None:
        assert _get_platform() == "unsupported"


class TestMacOSInstall:
    """Tests for macOS daemon install via launchd."""

    @patch("agent.core.daemon._get_platform", return_value="macos")
    @patch("agent.core.daemon._macos_plist_path")
    def test_install_creates_plist(
        self, mock_path: MagicMock, _mock_plat: MagicMock, tmp_path: Path
    ) -> None:
        plist_file = tmp_path / "com.agent.gateway.plist"
        mock_path.return_value = plist_file

        with patch("agent.core.daemon._macos_plist_content", return_value="<plist>content</plist>"):
            result = daemon_install()

        assert plist_file.exists()
        assert plist_file.read_text() == "<plist>content</plist>"
        assert "installed" in result.lower() or "Service installed" in result

    @patch("agent.core.daemon._get_platform", return_value="macos")
    @patch("agent.core.daemon._macos_plist_path")
    def test_install_already_exists(
        self, mock_path: MagicMock, _mock_plat: MagicMock, tmp_path: Path
    ) -> None:
        plist_file = tmp_path / "com.agent.gateway.plist"
        plist_file.write_text("existing")
        mock_path.return_value = plist_file

        result = daemon_install()
        assert "Already installed" in result

    @patch("agent.core.daemon._get_platform", return_value="unsupported")
    def test_install_unsupported_platform(self, _mock: MagicMock) -> None:
        result = daemon_install()
        assert "not supported" in result.lower()


class TestMacOSStop:
    """Tests for macOS daemon stop."""

    @patch("agent.core.daemon._get_platform", return_value="macos")
    @patch("agent.core.daemon._macos_plist_path")
    @patch("agent.core.daemon.subprocess.run")
    def test_stop_success(
        self,
        mock_run: MagicMock,
        mock_path: MagicMock,
        _mock_plat: MagicMock,
        tmp_path: Path,
    ) -> None:
        plist_file = tmp_path / "com.agent.gateway.plist"
        plist_file.write_text("<plist/>")
        mock_path.return_value = plist_file
        mock_run.return_value = MagicMock(returncode=0, stderr="")

        result = daemon_stop()
        assert "stopped" in result.lower()
        mock_run.assert_called_once()

    @patch("agent.core.daemon._get_platform", return_value="macos")
    @patch("agent.core.daemon._macos_plist_path")
    def test_stop_not_installed(
        self, mock_path: MagicMock, _mock_plat: MagicMock, tmp_path: Path
    ) -> None:
        plist_file = tmp_path / "nonexistent.plist"
        mock_path.return_value = plist_file

        result = daemon_stop()
        assert "not installed" in result.lower()

    @patch("agent.core.daemon._get_platform", return_value="macos")
    @patch("agent.core.daemon._macos_plist_path")
    @patch("agent.core.daemon.subprocess.run")
    def test_stop_failure_returns_error(
        self,
        mock_run: MagicMock,
        mock_path: MagicMock,
        _mock_plat: MagicMock,
        tmp_path: Path,
    ) -> None:
        plist_file = tmp_path / "com.agent.gateway.plist"
        plist_file.write_text("<plist/>")
        mock_path.return_value = plist_file
        mock_run.return_value = MagicMock(returncode=1, stderr="permission denied")

        result = daemon_stop()
        assert "Failed to stop" in result

    @patch("agent.core.daemon._get_platform", return_value="unsupported")
    def test_stop_unsupported(self, _mock: MagicMock) -> None:
        result = daemon_stop()
        assert "not supported" in result.lower()


class TestMacOSStart:
    """Tests for macOS daemon start."""

    @patch("agent.core.daemon._get_platform", return_value="macos")
    @patch("agent.core.daemon._macos_plist_path")
    @patch("agent.core.daemon.subprocess.run")
    def test_start_success(
        self,
        mock_run: MagicMock,
        mock_path: MagicMock,
        _mock_plat: MagicMock,
        tmp_path: Path,
    ) -> None:
        plist_file = tmp_path / "com.agent.gateway.plist"
        plist_file.write_text("<plist/>")
        mock_path.return_value = plist_file
        mock_run.return_value = MagicMock(returncode=0, stderr="")

        result = daemon_start()
        assert "started" in result.lower()

    @patch("agent.core.daemon._get_platform", return_value="macos")
    @patch("agent.core.daemon._macos_plist_path")
    def test_start_not_installed(
        self, mock_path: MagicMock, _mock_plat: MagicMock, tmp_path: Path
    ) -> None:
        plist_file = tmp_path / "nonexistent.plist"
        mock_path.return_value = plist_file

        result = daemon_start()
        assert "not installed" in result.lower() or "install" in result.lower()


class TestMacOSUninstall:
    """Tests for macOS daemon uninstall."""

    @patch("agent.core.daemon._get_platform", return_value="macos")
    @patch("agent.core.daemon._macos_plist_path")
    @patch("agent.core.daemon.subprocess.run")
    def test_uninstall_removes_plist(
        self,
        mock_run: MagicMock,
        mock_path: MagicMock,
        _mock_plat: MagicMock,
        tmp_path: Path,
    ) -> None:
        plist_file = tmp_path / "com.agent.gateway.plist"
        plist_file.write_text("<plist/>")
        mock_path.return_value = plist_file
        mock_run.return_value = MagicMock(returncode=0)

        result = daemon_uninstall()
        assert not plist_file.exists()
        assert "uninstalled" in result.lower()

    @patch("agent.core.daemon._get_platform", return_value="macos")
    @patch("agent.core.daemon._macos_plist_path")
    def test_uninstall_not_installed(
        self, mock_path: MagicMock, _mock_plat: MagicMock, tmp_path: Path
    ) -> None:
        plist_file = tmp_path / "nonexistent.plist"
        mock_path.return_value = plist_file

        result = daemon_uninstall()
        assert "not installed" in result.lower()


class TestMacOSStatusCheck:
    """Tests for macOS daemon status (is_running check)."""

    @patch("agent.core.daemon._get_platform", return_value="macos")
    @patch("agent.core.daemon._macos_plist_path")
    def test_status_not_installed(
        self, mock_path: MagicMock, _mock_plat: MagicMock, tmp_path: Path
    ) -> None:
        plist_file = tmp_path / "nonexistent.plist"
        mock_path.return_value = plist_file

        status = daemon_status()
        assert status.installed is False
        assert status.running is False

    @patch("agent.core.daemon._get_platform", return_value="macos")
    @patch("agent.core.daemon._macos_plist_path")
    @patch("agent.core.daemon._log_dir")
    @patch("agent.core.daemon.subprocess.run")
    def test_status_installed_and_running(
        self,
        mock_run: MagicMock,
        mock_log_dir: MagicMock,
        mock_path: MagicMock,
        _mock_plat: MagicMock,
        tmp_path: Path,
    ) -> None:
        plist_file = tmp_path / "com.agent.gateway.plist"
        plist_file.write_text("<plist/>")
        mock_path.return_value = plist_file
        mock_log_dir.return_value = tmp_path / "logs"

        # launchctl list returns PID in first column
        mock_run.return_value = MagicMock(
            returncode=0,
            stdout="99999\t0\tcom.agent.gateway\n",
        )

        status = daemon_status()
        assert status.installed is True
        assert status.running is True
        assert status.pid == 99999

    @patch("agent.core.daemon._get_platform", return_value="macos")
    @patch("agent.core.daemon._macos_plist_path")
    @patch("agent.core.daemon._log_dir")
    @patch("agent.core.daemon.subprocess.run")
    def test_status_installed_not_running(
        self,
        mock_run: MagicMock,
        mock_log_dir: MagicMock,
        mock_path: MagicMock,
        _mock_plat: MagicMock,
        tmp_path: Path,
    ) -> None:
        plist_file = tmp_path / "com.agent.gateway.plist"
        plist_file.write_text("<plist/>")
        mock_path.return_value = plist_file
        mock_log_dir.return_value = tmp_path / "logs"

        mock_run.return_value = MagicMock(returncode=1, stdout="")

        status = daemon_status()
        assert status.installed is True
        assert status.running is False
        assert status.pid is None

    @patch("agent.core.daemon._get_platform", return_value="unsupported")
    def test_status_unsupported_platform(self, _mock: MagicMock) -> None:
        status = daemon_status()
        assert status.installed is False
        assert status.running is False
