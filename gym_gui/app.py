from __future__ import annotations

"""Application entry-point helpers for manual smoke-testing."""

import json
import logging
import sys
from typing import Any

from gym_gui.config.settings import Settings, get_settings
from gym_gui.logging_config.logger import configure_logging
from gym_gui.services.bootstrap import bootstrap_default_services


def _format_settings(settings: Settings) -> str:
    payload: dict[str, Any] = {
        "qt_api": settings.qt_api,
        "gym_default_env": settings.gym_default_env,
        "gym_video_dir": str(settings.gym_video_dir) if settings.gym_video_dir else None,
        "enable_agent_autostart": settings.enable_agent_autostart,
        "log_level": settings.log_level,
        "use_gpu": settings.use_gpu,
        "default_control_mode": settings.default_control_mode.value,
        "agent_ids": settings.agent_ids,
    }
    return json.dumps(payload, indent=2)


def main() -> int:
    """Print the currently loaded settings and, if Qt is installed, show a stub window."""

    settings = get_settings()
    log_level = getattr(logging, settings.log_level.upper(), logging.INFO)
    configure_logging(level=log_level, stream=True, log_to_file=True)
    logger = logging.getLogger("gym_gui.app")
    logger.debug("Settings loaded: qt_api=%s default_env=%s", settings.qt_api, settings.gym_default_env)
    print("[gym_gui] Loaded settings:\n" + _format_settings(settings))

    bootstrap_default_services()

    try:
        from qtpy.QtWidgets import QApplication
        from gym_gui.ui.main_window import MainWindow
    except ImportError as exc:  # pragma: no cover - optional dependency
        print("[gym_gui] Qt bindings not available. Install qtpy and a Qt backend (PyQt5/PyQt6/PySide2/PySide6):", exc)
        return 0
    except Exception as exc:  # pragma: no cover - optional dependency
        print("[gym_gui] Unexpected error loading Qt bindings:", exc)
        return 0

    app = QApplication(sys.argv)
    app.setApplicationName("Gym GUI")
    window = MainWindow(settings)
    window.show()
    return app.exec()


if __name__ == "__main__":  # pragma: no cover - CLI convenience
    raise SystemExit(main())
