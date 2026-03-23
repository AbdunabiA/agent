"""Safety guardrails for tool execution.

Pre-execution validation of commands, file paths, and URLs.
Post-execution output sanitization and truncation.
"""

from __future__ import annotations

import ipaddress
import os
import re
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlparse

import structlog

from agent.config import ToolsConfig

logger = structlog.get_logger(__name__)


@dataclass
class GuardrailResult:
    """Result of a guardrail check."""

    allowed: bool
    reason: str = ""
    blocked_by: str | None = None  # Which pattern or rule blocked it


class Guardrails:
    """Safety checks before and after tool execution.

    Pre-execution: validate inputs, check blocklists.
    Post-execution: validate outputs, detect anomalies.
    """

    # Dangerous command patterns that should ALWAYS be blocked
    BLOCKED_PATTERNS: list[str] = [
        r"rm\s+(-\w*r\w*f|-\w*f\w*r)\w*\s+/(\s|$|;|&|\|)",  # rm -rf / variants
        r"rm\s+-rf\s+~/?$",  # rm -rf ~ or rm -rf ~/
        r"mkfs\.",  # Format filesystem
        r"dd\s+if=.*of=/dev/",  # Direct disk write
        r":\(\)\s*\{.*\|.*&",  # Fork bomb (classic)
        r"\w+\(\)\s*\{\s*\w+\s*\|\s*\w+\s*&",  # Fork bomb (named variant)
        r"chmod\s+-R\s+777\s+/",  # Recursive 777 on root
        r"curl.*\|\s*(ba)?sh",  # Pipe curl to shell
        r"wget.*\|\s*(ba)?sh",  # Pipe wget to shell
        r">\s*/dev/sd[a-z]",  # Overwrite disk device
        r"shutdown|reboot|poweroff",  # System power commands
        r"\b(passwd|chpasswd|usermod\s+-p)\b",  # Password change
        r"\buserdel\b|\buseradd\b",  # User management
    ]

    # Maximum output size before truncation
    MAX_OUTPUT_SIZE: int = 50 * 1024  # 50KB

    # ANSI escape code pattern
    ANSI_PATTERN = re.compile(r"\x1b\[[0-9;]*[a-zA-Z]")

    # Sensitive data patterns (for warning, not blocking)
    SENSITIVE_PATTERNS: list[tuple[str, str]] = [
        (r"(?:sk-|pk_live_|rk_live_)[a-zA-Z0-9]{20,}", "API key"),
        (r"(?:password|passwd|pwd)\s*[:=]\s*\S+", "password"),
        (r"(?:secret|token)\s*[:=]\s*\S+", "secret/token"),
    ]

    def __init__(self, config: ToolsConfig) -> None:
        self.config = config
        self._blocked_patterns = [re.compile(p) for p in self.BLOCKED_PATTERNS]
        self._custom_allowed: list[str] = config.shell.allowed_commands
        self._sensitive_patterns = [
            (re.compile(p, re.IGNORECASE), desc) for p, desc in self.SENSITIVE_PATTERNS
        ]

    def check_command(self, command: str) -> GuardrailResult:
        """Check if a shell command is safe to execute.

        Args:
            command: The shell command string.

        Returns:
            GuardrailResult with .allowed, .reason, .blocked_by.
        """
        # Check against blocked patterns
        for pattern in self._blocked_patterns:
            if pattern.search(command):
                return GuardrailResult(
                    allowed=False,
                    reason=f"Command matches blocked pattern: {pattern.pattern}",
                    blocked_by=pattern.pattern,
                )

        # If allowed_commands is not ["*"], check command starts with an allowed command
        if self._custom_allowed != ["*"]:
            cmd_name = command.strip().split()[0] if command.strip() else ""
            cmd_basename = os.path.basename(cmd_name)
            if cmd_basename not in self._custom_allowed and cmd_name not in self._custom_allowed:
                return GuardrailResult(
                    allowed=False,
                    reason=f"Command '{cmd_name}' is not in allowed commands list",
                    blocked_by="allowed_commands",
                )

        return GuardrailResult(allowed=True)

    def check_file_path(self, path: str, operation: str) -> GuardrailResult:
        """Check if a file operation is safe.

        Args:
            path: The file path being accessed.
            operation: "read", "write", "delete", "list".

        Returns:
            GuardrailResult with safety assessment.
        """
        try:
            import os

            resolved = Path(os.path.expanduser(path)).resolve()
        except (ValueError, OSError) as e:
            return GuardrailResult(
                allowed=False,
                reason=f"Invalid path: {e}",
                blocked_by="path_resolution",
            )

        # Check path is within filesystem root
        root = self.config.filesystem.root
        try:
            import os as _os

            root_resolved = Path(_os.path.expanduser(root)).resolve()
            if not resolved.is_relative_to(root_resolved):
                return GuardrailResult(
                    allowed=False,
                    reason=f"Path {path} is outside allowed root {root}",
                    blocked_by="filesystem_root",
                )
        except (ValueError, OSError):
            pass

        # Block write/delete to critical system paths
        if operation in ("write", "delete"):
            critical_paths = ["/etc", "/usr", "/bin", "/boot", "/sys", "/proc", "/sbin"]
            resolved_str_fwd = str(resolved).replace("\\", "/")
            # Check both resolved path and original path (original handles cross-platform
            # where e.g. "/etc/passwd" resolves to "C:\etc\passwd" on Windows)
            path_fwd = path.replace("\\", "/")
            for critical in critical_paths:
                if resolved_str_fwd.startswith(critical) or path_fwd.startswith(critical):
                    return GuardrailResult(
                        allowed=False,
                        reason=f"Cannot {operation} in system path: {critical}",
                        blocked_by="critical_path",
                    )

        # Warn about sensitive files (but allow)
        sensitive_files = [".ssh/id_rsa", ".ssh/id_ed25519", ".env"]
        resolved_str = str(resolved)
        for sensitive in sensitive_files:
            if resolved_str.endswith(sensitive):
                logger.warning(
                    "sensitive_file_access",
                    path=str(resolved),
                    operation=operation,
                )

        return GuardrailResult(allowed=True)

    def validate_output(self, output: str) -> str:
        """Post-execution output validation and sanitization.

        - Truncate if exceeds MAX_OUTPUT_SIZE.
        - Strip ANSI escape codes.
        - Detect and warn about sensitive data patterns.

        Args:
            output: Raw tool output string.

        Returns:
            Sanitized output string.
        """
        # Strip ANSI escape codes
        output = self.ANSI_PATTERN.sub("", output)

        # Detect sensitive data patterns (warn only)
        for pattern, desc in self._sensitive_patterns:
            if pattern.search(output):
                logger.warning("sensitive_data_in_output", type=desc)

        # Truncate if too large
        if len(output.encode("utf-8", errors="replace")) > self.MAX_OUTPUT_SIZE:
            original_size = len(output.encode("utf-8", errors="replace"))
            # Truncate by characters (rough approximation)
            truncated = output[: self.MAX_OUTPUT_SIZE]
            output = (
                f"[Output truncated: {original_size} bytes -> "
                f"{self.MAX_OUTPUT_SIZE} bytes. First 50KB shown]\n{truncated}"
            )

        return output

    def check_url(self, url: str) -> GuardrailResult:
        """Check if a URL is safe for HTTP requests.

        Blocks requests to private IPs, metadata endpoints, and file protocol.

        Args:
            url: The URL to check.

        Returns:
            GuardrailResult with safety assessment.
        """
        try:
            parsed = urlparse(url)
        except ValueError:
            return GuardrailResult(
                allowed=False,
                reason=f"Invalid URL: {url}",
                blocked_by="url_parse",
            )

        # Block file:// protocol
        if parsed.scheme == "file":
            return GuardrailResult(
                allowed=False,
                reason="file:// protocol is not allowed",
                blocked_by="file_protocol",
            )

        # Must be http or https
        if parsed.scheme not in ("http", "https"):
            return GuardrailResult(
                allowed=False,
                reason=f"Unsupported protocol: {parsed.scheme}",
                blocked_by="protocol",
            )

        # Check for private IPs
        hostname = parsed.hostname or ""

        # Block metadata endpoint
        if hostname == "169.254.169.254":
            return GuardrailResult(
                allowed=False,
                reason="Cloud metadata endpoint is blocked",
                blocked_by="metadata_endpoint",
            )

        # Check for private IP ranges
        try:
            ip = ipaddress.ip_address(hostname)
            if ip.is_private or ip.is_loopback or ip.is_link_local:
                return GuardrailResult(
                    allowed=False,
                    reason=f"Private/loopback IP address blocked: {hostname}",
                    blocked_by="private_ip",
                )
        except ValueError:
            # Not an IP address (hostname) — that's fine
            pass

        return GuardrailResult(allowed=True)
