#!/usr/bin/env python3
"""Fix Python 3.13+ editable installs on macOS.

Python 3.13 skips .pth files that have the macOS UF_HIDDEN flag set.
When a venv directory (e.g. .venv) is marked hidden by macOS, all files
inside inherit the flag, causing editable installs to silently break
(`import agent` fails with ModuleNotFoundError).

This script clears the UF_HIDDEN flag on the venv's site-packages
directory and all .pth files within it.

Usage:
    python scripts/fix_editable_install.py
"""

from __future__ import annotations

import os
import stat
import subprocess
import sys
from pathlib import Path


def main() -> None:
    # Find site-packages inside the current venv
    venv = os.environ.get("VIRTUAL_ENV")
    if not venv:
        print("No active virtualenv detected. Activate your venv first.")
        sys.exit(1)

    venv_path = Path(venv)
    if not venv_path.exists():
        print(f"Venv path does not exist: {venv_path}")
        sys.exit(1)

    if not hasattr(stat, "UF_HIDDEN"):
        print("Not macOS — no UF_HIDDEN flag to clear.")
        return

    # Check if the venv directory itself is hidden
    st = venv_path.stat()
    if not (st.st_flags & stat.UF_HIDDEN):
        print("Venv is not hidden — no fix needed.")
        return

    # Clear hidden flag recursively on the venv.
    # macOS sets UF_HIDDEN on dot-directories like .venv automatically.
    # Python 3.13+ skips .pth files inside hidden directories, breaking
    # editable installs.
    print(f"Clearing macOS hidden flag on {venv_path} ...")
    subprocess.run(
        ["chflags", "-R", "nohidden", str(venv_path)],
        check=True,
    )
    # Verify
    st2 = venv_path.stat()
    if st2.st_flags & stat.UF_HIDDEN:
        print("WARNING: chflags did not clear the flag. Try manually:")
        print(f"  chflags -R nohidden {venv_path}")
    else:
        print(f"Done. Fixed hidden flag for Python {sys.version.split()[0]}")


if __name__ == "__main__":
    main()
