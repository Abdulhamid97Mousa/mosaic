from __future__ import annotations

"""Application entry-point helpers for manual smoke-testing."""

# CRITICAL: Set Qt API BEFORE any other imports that might use Qt
import os
os.environ.setdefault("QT_API", "pyqt6")

import asyncio
import errno
import json
import logging
import sys
from dataclasses import replace
from functools import partial
from typing import Any

from gym_gui.config.settings import Settings, get_settings
from gym_gui.logging_config.logger import configure_logging
from gym_gui.logging_config.helpers import log_constant
from gym_gui.logging_config.log_constants import (
    LOG_RUNTIME_APP_DEBUG,
    LOG_RUNTIME_APP_INFO,
    LOG_RUNTIME_APP_WARNING,
)
# NOTE: bootstrap_default_services and TrainerDaemonHandle are imported inside main()
# to ensure Qt API is set before any Qt imports happen


LOGGER = logging.getLogger("gym_gui.app")
_log = partial(log_constant, LOGGER)


def _format_settings(settings: Settings) -> str:
    payload: dict[str, Any] = {
        "qt_api": settings.qt_api,
        "gym_default_env": settings.gym_default_env,
        "gym_video_dir": str(settings.gym_video_dir) if settings.gym_video_dir else None,
        "enable_agent_autostart": settings.enable_agent_autostart,
        "log_level": settings.log_level,
        "use_gpu": settings.use_gpu,
        "default_control_mode": settings.default_control_mode.value,
        "default_seed": settings.default_seed,
        "agent_ids": settings.agent_ids,
    }
    return json.dumps(payload, indent=2)


def main() -> int:
    """Print the currently loaded settings and, if Qt is installed, show a stub window."""

    os.environ.setdefault("QT_DEBUG_PLUGINS", "0")

    settings = replace(get_settings(), default_seed=1)
    log_level = getattr(logging, settings.log_level.upper(), logging.INFO)
    configure_logging(level=log_level, stream=False, log_to_file=True)
    _log(
        LOG_RUNTIME_APP_DEBUG,
        message="settings_loaded",
        extra={"qt_api": settings.qt_api, "default_env": settings.gym_default_env},
    )
    print("[gym_gui] Loaded settings:\n" + _format_settings(settings))

    try:
        from qtpy.QtWidgets import QApplication, QMessageBox
        from gym_gui.ui.main_window import MainWindow
    except ImportError as exc:  # pragma: no cover - optional dependency
        print("[gym_gui] Qt bindings not available. Install qtpy and a Qt backend (PyQt5/PyQt6/PySide2/PySide6):", exc)
        return 0
    except Exception as exc:  # pragma: no cover - optional dependency
        print("[gym_gui] Unexpected error loading Qt bindings:", exc)
        return 0

    app = QApplication(sys.argv)
    app.setApplicationName("Gym GUI")

    # Setup Qt-compatible asyncio event loop using qasync
    _setup_qasync_event_loop(app)

    # Import bootstrap AFTER Qt is initialized
    from gym_gui.services.bootstrap import bootstrap_default_services
    from gym_gui.services.trainer.launcher import TrainerDaemonHandle, TrainerDaemonLaunchError

    try:
        locator = bootstrap_default_services()
    except TrainerDaemonLaunchError as exc:
        QMessageBox.critical(None, "Trainer Daemon Error", str(exc))
        return 1
    except Exception as exc:  # pragma: no cover - defensive
        QMessageBox.critical(None, "Bootstrap Error", str(exc))
        return 1

    daemon_handle: TrainerDaemonHandle | None = locator.resolve(TrainerDaemonHandle) if locator else None

    window = MainWindow(settings)
    window.show()

    if daemon_handle:
        status_bar = window.statusBar()
        if status_bar is not None:
            if daemon_handle.reused:
                status_bar.showMessage("Connected to existing trainer daemon", 5000)
            else:
                status_bar.showMessage("Trainer daemon started automatically", 5000)

        def _stop_daemon() -> None:
            daemon_handle.stop()

        app.aboutToQuit.connect(_stop_daemon)

    return app.exec()


def _setup_qasync_event_loop(app: Any) -> None:
    """Setup Qt-compatible asyncio event loop using qasync.

    This allows asyncio and Qt to share the same event loop, preventing
    conflicts between the two event loop systems.
    """
    try:
        from qasync import QEventLoop
    except ImportError:
        _log(
            LOG_RUNTIME_APP_WARNING,
            message="qasync_missing_fallback",
        )
        _install_asyncio_exception_handler()
        return

    # Create Qt-compatible event loop
    loop = QEventLoop(app)
    asyncio.set_event_loop(loop)

    # Install exception handler
    def _handler(loop: asyncio.AbstractEventLoop, context: dict[str, Any]) -> None:
        exc = context.get("exception")
        if isinstance(exc, BlockingIOError) and getattr(exc, "errno", None) in {errno.EAGAIN, errno.EWOULDBLOCK}:
            _log(
                LOG_RUNTIME_APP_DEBUG,
                message="grpc_blocking_io_ignored",
                extra={"source": "qasync", "errno": getattr(exc, "errno", None)},
            )
            return
        loop.default_exception_handler(context)

    loop.set_exception_handler(_handler)
    _log(
        LOG_RUNTIME_APP_INFO,
        message="qasync_event_loop_initialized",
    )


def _install_asyncio_exception_handler() -> None:
    """Fallback: Install exception handler for separate asyncio loop."""
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

    def _handler(loop: asyncio.AbstractEventLoop, context: dict[str, Any]) -> None:
        exc = context.get("exception")
        if isinstance(exc, BlockingIOError) and getattr(exc, "errno", None) in {errno.EAGAIN, errno.EWOULDBLOCK}:
            _log(
                LOG_RUNTIME_APP_DEBUG,
                message="grpc_blocking_io_ignored",
                extra={"source": "asyncio", "errno": getattr(exc, "errno", None)},
            )
            return
        loop.default_exception_handler(context)

    loop.set_exception_handler(_handler)


if __name__ == "__main__":  # pragma: no cover - CLI convenience
    raise SystemExit(main())
