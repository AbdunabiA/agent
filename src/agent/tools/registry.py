"""Tool registry — @tool decorator + ToolRegistry for function-calling tools.

Provides a decorator that registers Python async functions as tools the LLM can call,
auto-generates JSON Schema from type hints, and manages tool lifecycle.
"""

from __future__ import annotations

import inspect
import types
from collections.abc import Callable
from dataclasses import dataclass
from enum import StrEnum
from typing import Any, Union, get_args, get_origin, get_type_hints

import structlog

logger = structlog.get_logger(__name__)


class ToolTier(StrEnum):
    """Permission tier for tool execution."""

    SAFE = "safe"  # Auto-approve always
    MODERATE = "moderate"  # Configurable: auto or ask
    DANGEROUS = "dangerous"  # Always requires user confirmation


class ToolError(Exception):
    """Base error for tool operations."""


class ToolTimeoutError(ToolError):
    """Tool execution timed out."""


class ToolNotFoundError(ToolError):
    """Tool not found in registry."""


class ToolPermissionError(ToolError):
    """Tool execution denied by permissions or path validation."""


@dataclass
class ToolDefinition:
    """Metadata about a registered tool."""

    name: str
    description: str
    tier: ToolTier
    parameters: dict[str, Any]  # JSON Schema for parameters
    function: Callable  # The actual async function
    category: str = "builtin"
    enabled: bool = True

    def to_llm_schema(self) -> dict[str, Any]:
        """Convert to the format LLMs expect for function calling.

        Returns a dict compatible with OpenAI/Anthropic tool format:
        {
            "type": "function",
            "function": {
                "name": "shell_exec",
                "description": "Run a shell command...",
                "parameters": { JSON Schema }
            }
        }
        """
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }


def _python_type_to_json_schema(python_type: Any) -> tuple[dict[str, Any], bool]:
    """Convert a Python type hint to JSON Schema.

    Args:
        python_type: The Python type annotation.

    Returns:
        Tuple of (schema_dict, is_optional).
    """
    is_optional = False

    # Handle Union types (str | None, Optional[str])
    origin = get_origin(python_type)
    if origin is Union or isinstance(python_type, types.UnionType):
        args = get_args(python_type)
        non_none_args = [a for a in args if a is not type(None)]
        if type(None) in args:
            is_optional = True
        if len(non_none_args) == 1 or len(non_none_args) > 1:
            python_type = non_none_args[0]
            origin = get_origin(python_type)

    # Map basic types
    type_map: dict[type, dict[str, str]] = {
        str: {"type": "string"},
        int: {"type": "integer"},
        float: {"type": "number"},
        bool: {"type": "boolean"},
        dict: {"type": "object"},
    }

    if python_type in type_map:
        return type_map[python_type], is_optional

    # Handle list / list[T]
    if origin is list or python_type is list:
        args = get_args(python_type)
        if args:
            item_schema, _ = _python_type_to_json_schema(args[0])
            return {"type": "array", "items": item_schema}, is_optional
        return {"type": "array"}, is_optional

    # Handle dict[K, V]
    if origin is dict:
        return {"type": "object"}, is_optional

    # Default fallback
    return {"type": "string"}, is_optional


def _generate_parameters_schema(func: Callable) -> dict[str, Any]:
    """Generate JSON Schema parameters from a function's type hints.

    Inspects the function's signature and type hints to produce a
    JSON Schema 'object' definition with properties and required fields.

    Args:
        func: The async function to inspect.

    Returns:
        JSON Schema dict for the function's parameters.
    """
    hints = get_type_hints(func)
    sig = inspect.signature(func)

    properties: dict[str, Any] = {}
    required: list[str] = []

    for param_name, param in sig.parameters.items():
        if param_name in ("self", "cls"):
            continue

        param_type = hints.get(param_name, str)

        # Skip return type
        if param_name == "return":
            continue

        schema, is_optional_type = _python_type_to_json_schema(param_type)
        schema["description"] = param_name
        properties[param_name] = schema

        # Parameter is required if it has no default value AND is not Optional type
        has_default = param.default is not inspect.Parameter.empty
        if not has_default and not is_optional_type:
            required.append(param_name)

    result: dict[str, Any] = {
        "type": "object",
        "properties": properties,
    }
    if required:
        result["required"] = required

    return result


class ToolRegistry:
    """Central registry for all available tools.

    Usage:
        registry = ToolRegistry()

        @registry.tool(name="my_tool", description="...", tier=ToolTier.SAFE)
        async def my_tool(param: str, count: int = 5) -> str:
            ...

        # Get all tool schemas for LLM
        schemas = registry.get_tool_schemas()

        # Look up a tool by name
        tool_def = registry.get_tool("my_tool")
    """

    def __init__(self) -> None:
        self._tools: dict[str, ToolDefinition] = {}
        self._generation: int = 0

    def tool(
        self,
        name: str,
        description: str,
        tier: ToolTier = ToolTier.MODERATE,
        category: str = "builtin",
    ) -> Callable:
        """Decorator to register a function as a tool.

        The function MUST be async.
        Parameters are extracted from type hints.
        JSON Schema is auto-generated from type hints.

        Args:
            name: Unique tool name.
            description: Human-readable description for the LLM.
            tier: Permission tier (SAFE, MODERATE, DANGEROUS).
            category: Tool category (builtin, skill, custom).

        Returns:
            Decorator function.

        Raises:
            ValueError: If function is not async or name is duplicate.
        """

        def decorator(func: Callable) -> Callable:
            # Validate function is async
            if not inspect.iscoroutinefunction(func):
                raise ValueError(f"Tool '{name}' must be an async function")

            # Check for duplicates
            if name in self._tools:
                raise ValueError(f"Tool '{name}' is already registered")

            # Generate JSON Schema from type hints
            parameters = _generate_parameters_schema(func)

            # Create ToolDefinition
            tool_def = ToolDefinition(
                name=name,
                description=description,
                tier=tier,
                parameters=parameters,
                function=func,
                category=category,
                enabled=True,
            )

            # Store in registry
            self._tools[name] = tool_def
            self._generation += 1
            logger.debug("tool_registered", name=name, tier=tier.value, category=category)

            return func

        return decorator

    def get_tool(self, name: str) -> ToolDefinition | None:
        """Look up a tool by name.

        Args:
            name: The tool name.

        Returns:
            ToolDefinition if found, None otherwise.
        """
        return self._tools.get(name)

    def get_tool_schemas(self, enabled_only: bool = True) -> list[dict[str, Any]]:
        """Get all tool schemas formatted for LLM function calling.

        Args:
            enabled_only: If True, only return schemas for enabled tools.

        Returns:
            List of tool schema dicts in LLM function-calling format.
        """
        schemas = []
        for tool_def in self._tools.values():
            if enabled_only and not tool_def.enabled:
                continue
            schemas.append(tool_def.to_llm_schema())
        return schemas

    def list_tools(self) -> list[ToolDefinition]:
        """List all registered tools.

        Returns:
            List of all ToolDefinition objects.
        """
        return list(self._tools.values())

    def enable_tool(self, name: str) -> None:
        """Enable a tool.

        Args:
            name: Tool name to enable.

        Raises:
            ToolNotFoundError: If tool doesn't exist.
        """
        tool_def = self._tools.get(name)
        if not tool_def:
            raise ToolNotFoundError(f"Tool '{name}' not found")
        tool_def.enabled = True
        self._generation += 1

    def disable_tool(self, name: str) -> None:
        """Disable a tool.

        Args:
            name: Tool name to disable.

        Raises:
            ToolNotFoundError: If tool doesn't exist.
        """
        tool_def = self._tools.get(name)
        if not tool_def:
            raise ToolNotFoundError(f"Tool '{name}' not found")
        tool_def.enabled = False
        self._generation += 1

    def unregister_tool(self, name: str) -> None:
        """Remove a tool from the registry.

        Args:
            name: Tool name to remove. Silently ignored if not found.
        """
        if name in self._tools:
            del self._tools[name]
            self._generation += 1


# Global registry instance
registry = ToolRegistry()


# Convenience decorator that uses the global registry
def tool(
    name: str,
    description: str,
    tier: ToolTier = ToolTier.MODERATE,
    category: str = "builtin",
) -> Callable:
    """Decorator shortcut using the global registry.

    Args:
        name: Unique tool name.
        description: Human-readable description.
        tier: Permission tier.
        category: Tool category.

    Returns:
        Decorator function.
    """
    return registry.tool(name=name, description=description, tier=tier, category=category)
