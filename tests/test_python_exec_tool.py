"""Tests for the Python execution tool."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

from agent.tools.builtins.python_exec import python_exec


class TestSimpleExecution:
    """Tests for basic code execution."""

    @patch("agent.tools.builtins.python_exec.asyncio.create_subprocess_exec")
    async def test_returns_stdout(self, mock_exec: AsyncMock) -> None:
        mock_proc = AsyncMock()
        mock_proc.communicate = AsyncMock(return_value=(b"hello world\n", b""))
        mock_proc.returncode = 0
        mock_exec.return_value = mock_proc

        result = await python_exec(code="print('hello world')")
        assert "hello world" in result

    @patch("agent.tools.builtins.python_exec.asyncio.create_subprocess_exec")
    async def test_multiline_output(self, mock_exec: AsyncMock) -> None:
        mock_proc = AsyncMock()
        mock_proc.communicate = AsyncMock(return_value=(b"line1\nline2\nline3\n", b""))
        mock_proc.returncode = 0
        mock_exec.return_value = mock_proc

        result = await python_exec(code="for i in range(3): print(f'line{i+1}')")
        assert "line1" in result
        assert "line2" in result
        assert "line3" in result

    @patch("agent.tools.builtins.python_exec.asyncio.create_subprocess_exec")
    async def test_no_output_returns_placeholder(self, mock_exec: AsyncMock) -> None:
        mock_proc = AsyncMock()
        mock_proc.communicate = AsyncMock(return_value=(b"", b""))
        mock_proc.returncode = 0
        mock_exec.return_value = mock_proc

        result = await python_exec(code="x = 1 + 1")
        assert "[No output]" in result


class TestOutputCapture:
    """Tests for stdout/stderr capture."""

    @patch("agent.tools.builtins.python_exec.asyncio.create_subprocess_exec")
    async def test_captures_stdout(self, mock_exec: AsyncMock) -> None:
        mock_proc = AsyncMock()
        mock_proc.communicate = AsyncMock(return_value=(b"printed output\n", b""))
        mock_proc.returncode = 0
        mock_exec.return_value = mock_proc

        result = await python_exec(code="print('printed output')")
        assert "printed output" in result

    @patch("agent.tools.builtins.python_exec.asyncio.create_subprocess_exec")
    async def test_captures_stderr_separately(self, mock_exec: AsyncMock) -> None:
        mock_proc = AsyncMock()
        mock_proc.communicate = AsyncMock(return_value=(b"", b"warning message\n"))
        mock_proc.returncode = 0
        mock_exec.return_value = mock_proc

        result = await python_exec(code="import sys; print('warning', file=sys.stderr)")
        assert "[STDERR]" in result
        assert "warning message" in result

    @patch("agent.tools.builtins.python_exec.asyncio.create_subprocess_exec")
    async def test_combines_stdout_and_stderr(self, mock_exec: AsyncMock) -> None:
        mock_proc = AsyncMock()
        mock_proc.communicate = AsyncMock(return_value=(b"out\n", b"err\n"))
        mock_proc.returncode = 0
        mock_exec.return_value = mock_proc

        result = await python_exec(code="...")
        assert "out" in result
        assert "[STDERR]" in result
        assert "err" in result


class TestSyntaxErrors:
    """Tests for syntax error handling."""

    @patch("agent.tools.builtins.python_exec.asyncio.create_subprocess_exec")
    async def test_syntax_error_in_stderr(self, mock_exec: AsyncMock) -> None:
        mock_proc = AsyncMock()
        mock_proc.communicate = AsyncMock(
            return_value=(
                b"",
                b'  File "/tmp/test.py", line 1\n    def\n       ^\nSyntaxError: invalid syntax\n',
            )
        )
        mock_proc.returncode = 1
        mock_exec.return_value = mock_proc

        result = await python_exec(code="def")
        assert "SyntaxError" in result
        assert "Exit code: 1" in result

    @patch("agent.tools.builtins.python_exec.asyncio.create_subprocess_exec")
    async def test_nonzero_exit_code_appended(self, mock_exec: AsyncMock) -> None:
        mock_proc = AsyncMock()
        mock_proc.communicate = AsyncMock(return_value=(b"", b"error\n"))
        mock_proc.returncode = 2
        mock_exec.return_value = mock_proc

        result = await python_exec(code="import sys; sys.exit(2)")
        assert "Exit code: 2" in result


class TestExceptionInCode:
    """Tests for runtime exceptions in executed code."""

    @patch("agent.tools.builtins.python_exec.asyncio.create_subprocess_exec")
    async def test_runtime_exception_captured(self, mock_exec: AsyncMock) -> None:
        mock_proc = AsyncMock()
        mock_proc.communicate = AsyncMock(
            return_value=(
                b"",
                b"Traceback (most recent call last):\n"
                b'  File "/tmp/test.py", line 1, in <module>\n'
                b"    raise ValueError('boom')\n"
                b"ValueError: boom\n",
            )
        )
        mock_proc.returncode = 1
        mock_exec.return_value = mock_proc

        result = await python_exec(code="raise ValueError('boom')")
        assert "ValueError" in result
        assert "boom" in result
        assert "Exit code: 1" in result

    @patch("agent.tools.builtins.python_exec.asyncio.create_subprocess_exec")
    async def test_timeout_kills_process(self, mock_exec: AsyncMock) -> None:
        mock_proc = AsyncMock()
        mock_proc.communicate = AsyncMock(side_effect=[TimeoutError(), (b"", b"")])
        mock_proc.kill = MagicMock()
        mock_exec.return_value = mock_proc

        result = await python_exec(code="import time; time.sleep(100)", timeout=1)

        assert "[ERROR]" in result
        assert "timed out" in result.lower()

    @patch("agent.tools.builtins.python_exec.asyncio.create_subprocess_exec")
    async def test_subprocess_creation_failure(self, mock_exec: AsyncMock) -> None:
        mock_exec.side_effect = OSError("No such file")

        result = await python_exec(code="print('hi')")
        assert "[ERROR]" in result
        assert "Failed to execute" in result
