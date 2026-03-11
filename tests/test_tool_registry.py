"""Tests for the tool registry."""

from __future__ import annotations

import pytest

from agent.tools.registry import (
    ToolNotFoundError,
    ToolRegistry,
    ToolTier,
    _generate_parameters_schema,
)


@pytest.fixture
def fresh_registry() -> ToolRegistry:
    """Create a fresh registry for each test."""
    return ToolRegistry()


class TestToolDecorator:
    """Tests for the @tool decorator."""

    def test_register_async_tool(self, fresh_registry: ToolRegistry) -> None:
        """Registering an async function should add it to the registry."""

        @fresh_registry.tool(name="test_tool", description="A test tool", tier=ToolTier.SAFE)
        async def test_tool(x: str) -> str:
            return x

        assert fresh_registry.get_tool("test_tool") is not None
        assert fresh_registry.get_tool("test_tool").name == "test_tool"
        assert fresh_registry.get_tool("test_tool").tier == ToolTier.SAFE

    def test_register_sync_function_raises(self, fresh_registry: ToolRegistry) -> None:
        """Registering a sync function should raise ValueError."""
        with pytest.raises(ValueError, match="must be an async function"):

            @fresh_registry.tool(name="bad_tool", description="Bad", tier=ToolTier.SAFE)
            def sync_tool(x: str) -> str:
                return x

    def test_duplicate_name_raises(self, fresh_registry: ToolRegistry) -> None:
        """Registering a tool with a duplicate name should raise ValueError."""

        @fresh_registry.tool(name="dup_tool", description="First", tier=ToolTier.SAFE)
        async def first(x: str) -> str:
            return x

        with pytest.raises(ValueError, match="already registered"):

            @fresh_registry.tool(name="dup_tool", description="Second", tier=ToolTier.SAFE)
            async def second(x: str) -> str:
                return x

    def test_tool_preserves_function(self, fresh_registry: ToolRegistry) -> None:
        """The decorator should return the original function."""

        @fresh_registry.tool(name="my_tool", description="Test", tier=ToolTier.SAFE)
        async def my_tool(x: str) -> str:
            return x

        # The decorated function should still be callable
        import asyncio

        result = asyncio.run(my_tool("hello"))
        assert result == "hello"


class TestJsonSchemaGeneration:
    """Tests for JSON Schema generation from type hints."""

    def test_str_type(self) -> None:
        async def f(x: str) -> str:
            return x

        schema = _generate_parameters_schema(f)
        assert schema["properties"]["x"]["type"] == "string"
        assert "x" in schema["required"]

    def test_int_type(self) -> None:
        async def f(x: int) -> str:
            return str(x)

        schema = _generate_parameters_schema(f)
        assert schema["properties"]["x"]["type"] == "integer"

    def test_float_type(self) -> None:
        async def f(x: float) -> str:
            return str(x)

        schema = _generate_parameters_schema(f)
        assert schema["properties"]["x"]["type"] == "number"

    def test_bool_type(self) -> None:
        async def f(x: bool) -> str:
            return str(x)

        schema = _generate_parameters_schema(f)
        assert schema["properties"]["x"]["type"] == "boolean"

    def test_list_str_type(self) -> None:
        async def f(x: list[str]) -> str:
            return ""

        schema = _generate_parameters_schema(f)
        assert schema["properties"]["x"]["type"] == "array"
        assert schema["properties"]["x"]["items"]["type"] == "string"

    def test_optional_not_required(self) -> None:
        async def f(x: str, y: str | None = None) -> str:
            return x

        schema = _generate_parameters_schema(f)
        assert "x" in schema["required"]
        assert "y" not in schema.get("required", [])

    def test_default_value_not_required(self) -> None:
        async def f(x: str, y: int = 5) -> str:
            return x

        schema = _generate_parameters_schema(f)
        assert "x" in schema["required"]
        assert "y" not in schema.get("required", [])

    def test_dict_type(self) -> None:
        async def f(x: dict) -> str:
            return ""

        schema = _generate_parameters_schema(f)
        assert schema["properties"]["x"]["type"] == "object"


class TestToolDefinition:
    """Tests for ToolDefinition."""

    def test_to_llm_schema(self, fresh_registry: ToolRegistry) -> None:
        """to_llm_schema should produce the expected format."""

        @fresh_registry.tool(
            name="shell_exec",
            description="Execute a shell command",
            tier=ToolTier.MODERATE,
        )
        async def shell_exec(command: str, timeout: int = 30) -> str:  # noqa: ASYNC109
            return ""

        tool_def = fresh_registry.get_tool("shell_exec")
        schema = tool_def.to_llm_schema()

        assert schema["type"] == "function"
        assert schema["function"]["name"] == "shell_exec"
        assert schema["function"]["description"] == "Execute a shell command"
        assert "command" in schema["function"]["parameters"]["properties"]
        assert "command" in schema["function"]["parameters"]["required"]
        assert "timeout" not in schema["function"]["parameters"].get("required", [])


class TestToolRegistry:
    """Tests for ToolRegistry methods."""

    def test_get_tool_schemas_enabled_only(self, fresh_registry: ToolRegistry) -> None:
        @fresh_registry.tool(name="t1", description="T1", tier=ToolTier.SAFE)
        async def t1() -> str:
            return ""

        @fresh_registry.tool(name="t2", description="T2", tier=ToolTier.SAFE)
        async def t2() -> str:
            return ""

        fresh_registry.disable_tool("t2")
        schemas = fresh_registry.get_tool_schemas(enabled_only=True)
        names = [s["function"]["name"] for s in schemas]
        assert "t1" in names
        assert "t2" not in names

    def test_get_tool_schemas_all(self, fresh_registry: ToolRegistry) -> None:
        @fresh_registry.tool(name="t1", description="T1", tier=ToolTier.SAFE)
        async def t1() -> str:
            return ""

        @fresh_registry.tool(name="t2", description="T2", tier=ToolTier.SAFE)
        async def t2() -> str:
            return ""

        fresh_registry.disable_tool("t2")
        schemas = fresh_registry.get_tool_schemas(enabled_only=False)
        assert len(schemas) == 2

    def test_enable_disable(self, fresh_registry: ToolRegistry) -> None:
        @fresh_registry.tool(name="t1", description="T1", tier=ToolTier.SAFE)
        async def t1() -> str:
            return ""

        assert fresh_registry.get_tool("t1").enabled is True
        fresh_registry.disable_tool("t1")
        assert fresh_registry.get_tool("t1").enabled is False
        fresh_registry.enable_tool("t1")
        assert fresh_registry.get_tool("t1").enabled is True

    def test_enable_nonexistent_raises(self, fresh_registry: ToolRegistry) -> None:
        with pytest.raises(ToolNotFoundError):
            fresh_registry.enable_tool("no_such_tool")

    def test_disable_nonexistent_raises(self, fresh_registry: ToolRegistry) -> None:
        with pytest.raises(ToolNotFoundError):
            fresh_registry.disable_tool("no_such_tool")

    def test_list_tools(self, fresh_registry: ToolRegistry) -> None:
        @fresh_registry.tool(name="t1", description="T1", tier=ToolTier.SAFE)
        async def t1() -> str:
            return ""

        tools = fresh_registry.list_tools()
        assert len(tools) == 1
        assert tools[0].name == "t1"

    def test_get_tool_not_found(self, fresh_registry: ToolRegistry) -> None:
        assert fresh_registry.get_tool("nonexistent") is None
