"""Tests for cross-platform system information tools."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path


class TestSystemInfo:
    """Tests for the system_info tool."""

    async def test_returns_os_info(self) -> None:
        from agent.tools.builtins.system import system_info

        result = await system_info()
        assert "OS:" in result
        assert "Platform:" in result
        assert "Hostname:" in result

    async def test_returns_cpu_info(self) -> None:
        from agent.tools.builtins.system import system_info

        result = await system_info()
        assert "CPU:" in result
        assert "cores" in result

    async def test_returns_ram_info(self) -> None:
        from agent.tools.builtins.system import system_info

        result = await system_info()
        assert "RAM:" in result
        assert "GB" in result

    async def test_returns_disk_info(self) -> None:
        from agent.tools.builtins.system import system_info

        result = await system_info()
        assert "Disk" in result

    async def test_returns_user_info(self) -> None:
        from agent.tools.builtins.system import system_info

        result = await system_info()
        assert "User:" in result
        assert "Home:" in result
        assert "CWD:" in result


class TestListDirectory:
    """Tests for the list_directory tool."""

    async def test_lists_files_with_details(self) -> None:
        from agent.tools.builtins.system import list_directory

        with tempfile.TemporaryDirectory() as tmpdir:
            Path(tmpdir, "file1.txt").write_text("hello")  # noqa: ASYNC240
            Path(tmpdir, "file2.py").write_text("print('hi')")  # noqa: ASYNC240
            os.makedirs(os.path.join(tmpdir, "subdir"))

            result = await list_directory(path=tmpdir)
            assert "file1.txt" in result
            assert "file2.py" in result
            assert "subdir" in result
            assert "items" in result  # total count

    async def test_permission_denied(self) -> None:
        from agent.tools.builtins.system import list_directory

        with tempfile.TemporaryDirectory() as tmpdir:
            restricted = os.path.join(tmpdir, "noperm")
            os.makedirs(restricted)
            # Create a path that doesn't exist to simulate error
            result = await list_directory(path=os.path.join(tmpdir, "nonexistent"))
            assert "ERROR" in result

    async def test_show_hidden_true(self) -> None:
        from agent.tools.builtins.system import list_directory

        with tempfile.TemporaryDirectory() as tmpdir:
            Path(tmpdir, ".hidden").write_text("secret")  # noqa: ASYNC240
            Path(tmpdir, "visible.txt").write_text("hello")  # noqa: ASYNC240

            # Without show_hidden
            result_no_hidden = await list_directory(path=tmpdir, show_hidden=False)
            assert ".hidden" not in result_no_hidden
            assert "visible.txt" in result_no_hidden

            # With show_hidden
            result_hidden = await list_directory(path=tmpdir, show_hidden=True)
            assert ".hidden" in result_hidden
            assert "visible.txt" in result_hidden

    async def test_sort_by_size(self) -> None:
        from agent.tools.builtins.system import list_directory

        with tempfile.TemporaryDirectory() as tmpdir:
            Path(tmpdir, "small.txt").write_text("a")  # noqa: ASYNC240
            Path(tmpdir, "big.txt").write_text("a" * 10000)  # noqa: ASYNC240

            result = await list_directory(path=tmpdir, sort_by="size")
            # big.txt should appear before small.txt when sorted by size desc
            big_pos = result.index("big.txt")
            small_pos = result.index("small.txt")
            assert big_pos < small_pos

    async def test_sort_by_name(self) -> None:
        from agent.tools.builtins.system import list_directory

        with tempfile.TemporaryDirectory() as tmpdir:
            Path(tmpdir, "beta.txt").write_text("b")  # noqa: ASYNC240
            Path(tmpdir, "alpha.txt").write_text("a")  # noqa: ASYNC240

            result = await list_directory(path=tmpdir, sort_by="name")
            alpha_pos = result.index("alpha.txt")
            beta_pos = result.index("beta.txt")
            assert alpha_pos < beta_pos

    async def test_sort_by_modified(self) -> None:
        from agent.tools.builtins.system import list_directory

        with tempfile.TemporaryDirectory() as tmpdir:
            Path(tmpdir, "file1.txt").write_text("a")  # noqa: ASYNC240
            Path(tmpdir, "file2.txt").write_text("b")  # noqa: ASYNC240

            result = await list_directory(path=tmpdir, sort_by="modified")
            assert "file1.txt" in result
            assert "file2.txt" in result

    async def test_not_a_directory(self) -> None:
        from agent.tools.builtins.system import list_directory

        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = os.path.join(tmpdir, "file.txt")
            Path(filepath).write_text("hello")  # noqa: ASYNC240

            result = await list_directory(path=filepath)
            assert "ERROR" in result
            assert "Not a directory" in result


class TestFindFiles:
    """Tests for the find_files tool."""

    async def test_find_by_pattern(self) -> None:
        from agent.tools.builtins.system import find_files

        with tempfile.TemporaryDirectory() as tmpdir:
            Path(tmpdir, "test.py").write_text("# python")  # noqa: ASYNC240
            Path(tmpdir, "test.txt").write_text("text")  # noqa: ASYNC240
            Path(tmpdir, "other.py").write_text("# other")  # noqa: ASYNC240

            result = await find_files(directory=tmpdir, pattern="*.py")
            assert "test.py" in result
            assert "other.py" in result
            assert "test.txt" not in result

    async def test_find_by_content(self) -> None:
        from agent.tools.builtins.system import find_files

        with tempfile.TemporaryDirectory() as tmpdir:
            Path(tmpdir, "match.txt").write_text("This has the NEEDLE in it")  # noqa: ASYNC240
            Path(tmpdir, "nomatch.txt").write_text("Nothing here")  # noqa: ASYNC240

            result = await find_files(directory=tmpdir, contains="needle")
            assert "match.txt" in result
            assert "nomatch.txt" not in result

    async def test_find_by_extension(self) -> None:
        from agent.tools.builtins.system import find_files

        with tempfile.TemporaryDirectory() as tmpdir:
            Path(tmpdir, "code.py").write_text("python")  # noqa: ASYNC240
            Path(tmpdir, "style.css").write_text("css")  # noqa: ASYNC240

            result = await find_files(directory=tmpdir, extension=".py")
            assert "code.py" in result
            assert "style.css" not in result

    async def test_respects_max_depth(self) -> None:
        from agent.tools.builtins.system import find_files

        with tempfile.TemporaryDirectory() as tmpdir:
            Path(tmpdir, "top.py").write_text("top")  # noqa: ASYNC240
            deep = os.path.join(tmpdir, "a", "b", "c", "d", "e", "f")
            os.makedirs(deep)
            Path(deep, "deep.py").write_text("deep")  # noqa: ASYNC240

            result = await find_files(directory=tmpdir, pattern="*.py", max_depth=2)
            assert "top.py" in result
            assert "deep.py" not in result

    async def test_respects_max_results(self) -> None:
        from agent.tools.builtins.system import find_files

        with tempfile.TemporaryDirectory() as tmpdir:
            for i in range(20):
                Path(tmpdir, f"file{i}.txt").write_text(f"content {i}")  # noqa: ASYNC240

            result = await find_files(directory=tmpdir, max_results=5)
            assert "Found 5 files" in result

    async def test_nonexistent_directory(self) -> None:
        from agent.tools.builtins.system import find_files

        result = await find_files(directory="/nonexistent_dir_12345")
        assert "ERROR" in result

    async def test_no_matches(self) -> None:
        from agent.tools.builtins.system import find_files

        with tempfile.TemporaryDirectory() as tmpdir:
            Path(tmpdir, "file.txt").write_text("content")  # noqa: ASYNC240

            result = await find_files(directory=tmpdir, pattern="*.xyz")
            assert "No files found" in result


class TestDiskUsage:
    """Tests for the disk_usage tool."""

    async def test_shows_partitions(self) -> None:
        from agent.tools.builtins.system import disk_usage

        result = await disk_usage()
        assert "Disk Usage:" in result
        assert "GB" in result

    async def test_shows_usage_percent(self) -> None:
        from agent.tools.builtins.system import disk_usage

        result = await disk_usage()
        assert "%" in result


class TestRunningProcesses:
    """Tests for the running_processes tool."""

    async def test_lists_processes(self) -> None:
        from agent.tools.builtins.system import running_processes

        result = await running_processes()
        assert "PID" in result
        assert "CPU%" in result
        assert "MEM%" in result

    async def test_sort_by_memory(self) -> None:
        from agent.tools.builtins.system import running_processes

        result = await running_processes(sort_by="memory")
        assert "PID" in result

    async def test_sort_by_cpu(self) -> None:
        from agent.tools.builtins.system import running_processes

        result = await running_processes(sort_by="cpu")
        assert "PID" in result

    async def test_sort_by_name(self) -> None:
        from agent.tools.builtins.system import running_processes

        result = await running_processes(sort_by="name")
        assert "PID" in result

    async def test_limit(self) -> None:
        from agent.tools.builtins.system import running_processes

        result = await running_processes(limit=5)
        # Header + separator + up to 5 processes = max 7 lines
        lines = [line for line in result.strip().split("\n") if line.strip()]
        assert len(lines) <= 7


class TestEnvironmentVars:
    """Tests for the environment_vars tool."""

    async def test_get_specific_var(self) -> None:
        from agent.tools.builtins.system import environment_vars

        os.environ["TEST_AGENT_VAR"] = "test_value_123"
        try:
            result = await environment_vars(name="TEST_AGENT_VAR")
            assert "test_value_123" in result
        finally:
            del os.environ["TEST_AGENT_VAR"]

    async def test_missing_var(self) -> None:
        from agent.tools.builtins.system import environment_vars

        result = await environment_vars(name="NONEXISTENT_VAR_XYZ_123")
        assert "not set" in result

    async def test_masks_sensitive_values(self) -> None:
        from agent.tools.builtins.system import environment_vars

        os.environ["MY_SECRET_KEY"] = "super_secret_value"
        try:
            result = await environment_vars(name="MY_SECRET_KEY")
            assert "super_secret_value" not in result
            assert "masked" in result.lower() or "*******" in result
        finally:
            del os.environ["MY_SECRET_KEY"]

    async def test_masks_sensitive_in_list(self) -> None:
        from agent.tools.builtins.system import environment_vars

        os.environ["TEST_API_TOKEN"] = "tok_12345"
        try:
            result = await environment_vars(filter_prefix="TEST_API")
            assert "tok_12345" not in result
            assert "*******" in result
        finally:
            del os.environ["TEST_API_TOKEN"]

    async def test_filter_prefix(self) -> None:
        from agent.tools.builtins.system import environment_vars

        os.environ["AGENTTEST_FOO"] = "bar"
        os.environ["AGENTTEST_BAZ"] = "qux"
        try:
            result = await environment_vars(filter_prefix="AGENTTEST")
            assert "AGENTTEST_FOO" in result
            assert "AGENTTEST_BAZ" in result
        finally:
            del os.environ["AGENTTEST_FOO"]
            del os.environ["AGENTTEST_BAZ"]

    async def test_list_all(self) -> None:
        from agent.tools.builtins.system import environment_vars

        result = await environment_vars()
        assert "Environment Variables" in result
