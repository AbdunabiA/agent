"""Auto-install missing dependencies on startup.

Checks for required and optional packages, installs any that are missing
using pip in the current environment.
"""

from __future__ import annotations

import importlib
import subprocess
import sys

import structlog

logger = structlog.get_logger(__name__)

# Map: import_name -> pip_package_name
# Core deps should already be installed; these are the ones that commonly fail.
OPTIONAL_DEPS: dict[str, str] = {
    "playwright": "playwright",
    "PIL": "Pillow",
    "pyautogui": "pyautogui",
    "pyperclip": "pyperclip",
    "chromadb": "chromadb",
    "aiogram": "aiogram",
    "websockets": "websockets",
    "claude_code_sdk": "claude-code-sdk",
    "mcp": "mcp",
}

# Packages that need a post-install step (command to run after pip install)
POST_INSTALL: dict[str, list[str]] = {
    "playwright": ["playwright", "install", "chromium"],
}


def check_missing() -> list[str]:
    """Return list of pip package names that are not importable."""
    missing: list[str] = []
    for import_name, pip_name in OPTIONAL_DEPS.items():
        try:
            importlib.import_module(import_name)
        except BaseException:
            missing.append(pip_name)
    return missing


def install_packages(packages: list[str]) -> bool:
    """Install packages via pip. Returns True if all succeeded."""
    if not packages:
        return True

    logger.info("auto_installing_dependencies", packages=packages)
    try:
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", *packages],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
        )
        logger.info("dependencies_installed", packages=packages)
    except subprocess.CalledProcessError as e:
        logger.error(
            "dependency_install_failed",
            packages=packages,
            error=e.stderr.decode() if e.stderr else str(e),
        )
        return False

    # Run post-install steps
    for pkg in packages:
        if pkg in POST_INSTALL:
            cmd = POST_INSTALL[pkg]
            logger.info("running_post_install", package=pkg, command=cmd)
            try:
                subprocess.check_call(
                    [sys.executable, "-m", *cmd],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.PIPE,
                )
                logger.info("post_install_complete", package=pkg)
            except subprocess.CalledProcessError as e:
                logger.warning(
                    "post_install_failed",
                    package=pkg,
                    error=e.stderr.decode() if e.stderr else str(e),
                )

    return True


def ensure_dependencies() -> None:
    """Check and auto-install any missing optional dependencies."""
    missing = check_missing()
    if not missing:
        logger.debug("all_optional_deps_present")
        return

    logger.info("missing_optional_dependencies", packages=missing)
    install_packages(missing)
