"""MCP server building for Claude SDK service."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import structlog

if TYPE_CHECKING:
    from agent.llm.claude_sdk._core import ClaudeSDKService

logger = structlog.get_logger(__name__)


def _build_mcp_server(
    self: ClaudeSDKService,
    registry: Any | None = None,
    tool_executor: Any | None = None,
    session_id: str = "sdk",
) -> tuple[Any, list[str]] | None:
    """Build an in-process MCP server exposing agent tools to the SDK.

    Creates SDK-compatible tool wrappers for each enabled tool in the
    registry, bundles them into an MCP server, and returns the server
    config + list of tool names.

    When ``tool_executor`` is provided, tool calls are routed through
    the executor for permission checks, guardrails, and audit logging.
    Otherwise falls back to direct function invocation.

    Args:
        registry: Optional tool registry override. Falls back to
            ``self.tool_registry`` when not provided.
        tool_executor: Optional ToolExecutor for safety routing.
        session_id: Session ID for audit logging when using executor.

    Returns:
        Tuple of (McpSdkServerConfig, tool_name_list) or None if no tools.
    """
    effective_registry = registry or self.tool_registry
    if not effective_registry:
        return None

    from claude_code_sdk import create_sdk_mcp_server
    from claude_code_sdk import tool as sdk_tool

    tools = effective_registry.list_tools()
    enabled_tools = [t for t in tools if t.enabled]
    if not enabled_tools:
        return None

    sdk_tools = []
    tool_names = []

    for tool_def in enabled_tools:
        # Capture tool_def in closure via default arg
        def _make_handler(
            td: Any,
            executor: Any = tool_executor,
            sid: str = session_id,
        ) -> Any:
            async def handler(args: dict[str, Any]) -> dict[str, Any]:
                from agent.tools.executor import MultimodalToolOutput

                # Route through executor for permission/audit/guardrails
                if executor is not None:
                    from uuid import uuid4

                    from agent.core.session import ToolCall

                    tc = ToolCall(
                        id=str(uuid4())[:8],
                        name=td.name,
                        arguments=args,
                    )
                    result = await executor.execute(
                        tool_call=tc,
                        session_id=sid,
                        trigger="sdk_mcp",
                    )
                    if result.images:
                        content_blocks: list[dict[str, Any]] = [
                            {"type": "text", "text": result.output},
                        ]
                        for img in result.images:
                            content_blocks.append(
                                {
                                    "type": "image",
                                    "data": img.base64_data,
                                    "mimeType": img.media_type,
                                }
                            )
                        return {"content": content_blocks}
                    return {
                        "content": [{"type": "text", "text": result.output}],
                    }

                # Direct call fallback (no executor)
                try:
                    result = await td.function(**args)
                    if isinstance(result, MultimodalToolOutput):
                        content_blocks = [
                            {"type": "text", "text": result.text},
                        ]
                        for img in result.images:
                            content_blocks.append(
                                {
                                    "type": "image",
                                    "data": img.base64_data,
                                    "mimeType": img.media_type,
                                }
                            )
                        return {"content": content_blocks}
                    text = str(result) if result is not None else ""
                except Exception as e:
                    text = f"Error: {e}"
                return {
                    "content": [{"type": "text", "text": text}],
                }

            return handler

        wrapped = sdk_tool(
            tool_def.name,
            tool_def.description,
            tool_def.parameters,
        )(_make_handler(tool_def))

        sdk_tools.append(wrapped)
        tool_names.append(tool_def.name)

    server = create_sdk_mcp_server(
        name="agent-tools",
        version="1.0.0",
        tools=sdk_tools,
    )

    logger.info("sdk_mcp_server_built", tool_count=len(sdk_tools))
    return server, tool_names
