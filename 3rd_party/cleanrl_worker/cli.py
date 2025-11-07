"""Entry point wrapper for CleanRL worker CLI."""

from .MOSAIC_CLEANRL_WORKER.cli import main  # noqa: F401


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
