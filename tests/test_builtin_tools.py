"""Tests for built-in tools."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest


class TestShellExec:
    """Tests for the shell_exec tool."""

    async def test_simple_command(self) -> None:
        from agent.tools.builtins.shell import shell_exec

        result = await shell_exec(command="echo hello")
        assert "hello" in result

    async def test_captures_stderr(self) -> None:
        from agent.tools.builtins.shell import shell_exec

        result = await shell_exec(command="echo error >&2")
        assert "error" in result

    async def test_nonzero_exit_code(self) -> None:
        from agent.tools.builtins.shell import shell_exec

        result = await shell_exec(command="exit 42")
        assert "Exit code: 42" in result

    async def test_timeout(self) -> None:
        from agent.tools.builtins.shell import shell_exec

        result = await shell_exec(command="sleep 30", timeout=1)
        assert "timed out" in result.lower()

    async def test_working_directory(self) -> None:
        from agent.tools.builtins.shell import shell_exec

        with tempfile.TemporaryDirectory() as tmpdir:
            result = await shell_exec(command="pwd", working_dir=tmpdir)
            # On Windows with MSYS, the path may be different
            assert tmpdir.replace("\\", "/") in result.replace("\\", "/") or os.path.basename(
                tmpdir
            ) in result


class TestFileRead:
    """Tests for the file_read tool."""

    async def test_read_existing_file(self) -> None:
        from agent.tools.builtins.filesystem import file_read

        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = os.path.join(tmpdir, "test.txt")
            Path(filepath).write_text("hello world")  # noqa: ASYNC240
            result = await file_read(path=filepath)
            assert "hello world" in result

    async def test_read_missing_file(self) -> None:
        from agent.tools.builtins.filesystem import file_read

        with tempfile.TemporaryDirectory() as tmpdir:
            missing = os.path.join(tmpdir, "nonexistent_file.txt")
            result = await file_read(path=missing)
            assert "not found" in result.lower() or "error" in result.lower()

    async def test_read_max_lines(self) -> None:
        from agent.tools.builtins.filesystem import file_read

        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = os.path.join(tmpdir, "lines.txt")
            with open(filepath, "w") as f:  # noqa: ASYNC230
                for i in range(100):
                    f.write(f"line {i}\n")
            result = await file_read(path=filepath, max_lines=5)
            assert "line 0" in result
            assert "line 4" in result
            assert "5" in result  # "Showing first 5 of..."


class TestFileWrite:
    """Tests for the file_write tool."""

    async def test_create_file(self) -> None:
        from agent.tools.builtins.filesystem import file_write

        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = os.path.join(tmpdir, "test.txt")
            with patch("agent.tools.builtins.filesystem._get_fs_config",
                        return_value=("/", tmpdir, [])):
                result = await file_write(path=filepath, content="hello")
            assert "written" in result.lower() or "Written" in result
            assert Path(filepath).read_text() == "hello"  # noqa: ASYNC240

    async def test_create_parent_dirs(self) -> None:
        from agent.tools.builtins.filesystem import file_write

        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = os.path.join(tmpdir, "sub", "dir", "test.txt")
            with patch("agent.tools.builtins.filesystem._get_fs_config",
                        return_value=("/", tmpdir, [])):
                await file_write(path=filepath, content="nested")
            assert Path(filepath).exists()  # noqa: ASYNC240
            assert Path(filepath).read_text() == "nested"  # noqa: ASYNC240

    async def test_append_mode(self) -> None:
        from agent.tools.builtins.filesystem import file_write

        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = os.path.join(tmpdir, "append.txt")
            Path(filepath).write_text("first")  # noqa: ASYNC240
            with patch("agent.tools.builtins.filesystem._get_fs_config",
                        return_value=("/", tmpdir, [])):
                await file_write(path=filepath, content=" second", append=True)
            assert Path(filepath).read_text() == "first second"  # noqa: ASYNC240


class TestFileList:
    """Tests for the file_list tool."""

    async def test_list_directory(self) -> None:
        from agent.tools.builtins.filesystem import file_list

        with tempfile.TemporaryDirectory() as tmpdir:
            Path(tmpdir, "file1.txt").write_text("a")  # noqa: ASYNC240
            Path(tmpdir, "file2.txt").write_text("bb")  # noqa: ASYNC240
            os.makedirs(os.path.join(tmpdir, "subdir"))

            result = await file_list(path=tmpdir)
            assert "file1.txt" in result
            assert "file2.txt" in result
            assert "subdir" in result

    async def test_list_nonexistent(self) -> None:
        from agent.tools.builtins.filesystem import file_list

        with tempfile.TemporaryDirectory() as tmpdir:
            missing = os.path.join(tmpdir, "nonexistent_dir")
            result = await file_list(path=missing)
            assert "not found" in result.lower() or "error" in result.lower()

    async def test_max_depth(self) -> None:
        from agent.tools.builtins.filesystem import file_list

        with tempfile.TemporaryDirectory() as tmpdir:
            deep_path = os.path.join(tmpdir, "a", "b", "c")
            os.makedirs(deep_path)
            Path(deep_path, "deep.txt").write_text("deep")  # noqa: ASYNC240

            # Depth 1 should not show deep.txt
            result = await file_list(path=tmpdir, max_depth=1)
            assert "deep.txt" not in result


class TestPathValidation:
    """Tests for path validation security."""

    async def test_path_traversal_blocked(self) -> None:
        from agent.tools.builtins.filesystem import _validate_path
        from agent.tools.registry import ToolPermissionError

        with (
            tempfile.TemporaryDirectory() as tmpdir,
            pytest.raises(ToolPermissionError, match="outside allowed root"),
        ):
            _validate_path("../../etc/passwd", tmpdir)


class TestPythonExec:
    """Tests for the python_exec tool."""

    async def test_simple_code(self) -> None:
        from agent.tools.builtins.python_exec import python_exec

        result = await python_exec(code="print('hello from python')")
        assert "hello from python" in result

    async def test_error_handling(self) -> None:
        from agent.tools.builtins.python_exec import python_exec

        result = await python_exec(code="raise ValueError('test error')")
        assert "ValueError" in result or "test error" in result

    async def test_timeout(self) -> None:
        from agent.tools.builtins.python_exec import python_exec

        result = await python_exec(code="import time; time.sleep(30)", timeout=1)
        assert "timed out" in result.lower()


class TestHttpRequest:
    """Tests for the http_request tool."""

    async def test_invalid_url(self) -> None:
        from agent.tools.builtins.http import http_request

        result = await http_request(url="not_a_url")
        assert "error" in result.lower()

    async def test_invalid_method(self) -> None:
        from agent.tools.builtins.http import http_request

        result = await http_request(url="https://example.com", method="INVALID")
        assert "error" in result.lower()

    @patch("agent.tools.builtins.http.httpx.AsyncClient")
    async def test_get_request(self, mock_client_class: AsyncMock) -> None:
        from agent.tools.builtins.http import http_request

        mock_response = AsyncMock()
        mock_response.status_code = 200
        mock_response.reason_phrase = "OK"
        mock_response.headers = {"content-type": "application/json"}
        mock_response.text = '{"result": "ok"}'

        mock_client = AsyncMock()
        mock_client.request = AsyncMock(return_value=mock_response)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        mock_client_class.return_value = mock_client

        result = await http_request(url="https://api.example.com/data")
        assert "200" in result
