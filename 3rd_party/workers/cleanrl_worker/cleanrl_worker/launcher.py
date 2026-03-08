"""Launch CleanRL modules with vendored dependency bootstrapping."""

from __future__ import annotations

import importlib
import subprocess
import sys
from pathlib import Path


def _bootstrap() -> None:
    """Ensure cleanrl and cleanrl_utils are importable.

    With proper package installation (pip install -e), these are already
    available as top-level packages. This bootstrap is a no-op but kept
    for compatibility.
    """
    # cleanrl and cleanrl_utils are now proper top-level packages
    # installed via pyproject.toml, no aliasing needed
    pass


def _run_subprocess(module_name: str, module_args: list[str]) -> int:
    cmd = [sys.executable, "-m", module_name, *module_args]
    return subprocess.call(cmd)


def main() -> None:
    if len(sys.argv) < 2:
        raise SystemExit("Usage: python -m cleanrl_worker.launcher <module> [args...]")

    module_name = sys.argv[1]
    module_args = sys.argv[2:]

    _bootstrap()

    # Prefer an explicit main() if the module exposes one.
    try:
        module = importlib.import_module(module_name)
        entry = getattr(module, "main", None)
        if callable(entry):
            sys.argv = [module_name, *module_args]
            entry()
            return
    except Exception:
        pass

    raise SystemExit(_run_subprocess(module_name, module_args))


if __name__ == "__main__":  # pragma: no cover
    main()
