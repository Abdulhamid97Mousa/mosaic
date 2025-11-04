"""Aggregate Gymnasium documentation snippets from submodules."""

from __future__ import annotations

from importlib import import_module

__all__: list[str] = []


def _export_from(module_name: str) -> None:
    module = import_module(f"{__name__}.{module_name}")
    for name in getattr(module, "__all__", ()):  # type: ignore[arg-type]
        globals()[name] = getattr(module, name)
        __all__.append(name)


for _submodule in ("ToyText", "Box2D", "MuJuCo"):
    _export_from(_submodule)

del _export_from, _submodule
