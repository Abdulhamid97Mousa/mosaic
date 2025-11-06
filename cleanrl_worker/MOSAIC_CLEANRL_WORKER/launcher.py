"""Launch CleanRL modules with vendored dependency bootstrapping."""

from __future__ import annotations

import importlib
import runpy
import sys
from pathlib import Path


def _bootstrap() -> None:
    vendor_root = Path(__file__).resolve().parent.parent
    vendored_path = str(vendor_root)
    if vendored_path not in sys.path:
        sys.path.insert(0, vendored_path)

    cleanrl_pkg = importlib.import_module("cleanrl_worker.cleanrl")
    sys.modules.setdefault("cleanrl", cleanrl_pkg)

    cleanrl_utils_pkg = importlib.import_module("cleanrl_worker.cleanrl_utils")
    sys.modules.setdefault("cleanrl_utils", cleanrl_utils_pkg)


def main() -> None:
    if len(sys.argv) < 2:
        raise SystemExit("Usage: python -m cleanrl_worker.MOSAIC_CLEANRL_WORKER.launcher <module> [args...]")

    module_name = sys.argv[1]
    module_args = sys.argv[2:]

    _bootstrap()

    sys.argv = [module_name, *module_args]
    runpy.run_module(module_name, run_name="__main__", alter_sys=True)


if __name__ == "__main__":  # pragma: no cover
    main()
