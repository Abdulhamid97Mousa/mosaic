"""Launch CleanRL modules with vendored dependency bootstrapping."""

from __future__ import annotations

import importlib
import subprocess
import sys
from pathlib import Path


def _bootstrap() -> None:
    vendor_root = Path(__file__).resolve().parent.parent
    vendored_path = str(vendor_root)
    if vendored_path not in sys.path:
        sys.path.insert(0, vendored_path)

    # Alias vendored packages so modules can import `cleanrl` directly.
    cleanrl_pkg = importlib.import_module("cleanrl_worker.cleanrl")
    sys.modules.setdefault("cleanrl", cleanrl_pkg)

    cleanrl_utils_pkg = importlib.import_module("cleanrl_worker.cleanrl_utils")
    sys.modules.setdefault("cleanrl_utils", cleanrl_utils_pkg)


def _run_subprocess(module_name: str, module_args: list[str]) -> int:
    cmd = [sys.executable, "-m", module_name, *module_args]
    return subprocess.call(cmd)


def main() -> None:
    if len(sys.argv) < 2:
        raise SystemExit("Usage: python -m cleanrl_worker.MOSAIC_CLEANRL_WORKER.launcher <module> [args...]")

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
