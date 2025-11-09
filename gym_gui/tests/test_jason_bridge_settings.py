from __future__ import annotations

import os

from gym_gui.config.settings import reload_settings, get_settings
from gym_gui.services.bootstrap import bootstrap_default_services
from gym_gui.services.jason_bridge import JasonBridgeServer
from gym_gui.services.service_locator import get_service_locator


def test_settings_env_overrides_bridge_fields(monkeypatch):
    monkeypatch.setenv("JASON_BRIDGE_ENABLED", "1")
    monkeypatch.setenv("JASON_BRIDGE_HOST", "0.0.0.0")
    monkeypatch.setenv("JASON_BRIDGE_PORT", "50666")
    # Reload settings after env variables are set
    settings = reload_settings()
    assert settings.jason_bridge_enabled is True
    assert settings.jason_bridge_host == "0.0.0.0"
    assert settings.jason_bridge_port == 50666


def test_bootstrap_uses_settings_for_bridge(monkeypatch):
    monkeypatch.setenv("JASON_BRIDGE_ENABLED", "1")
    monkeypatch.setenv("JASON_BRIDGE_PORT", "50777")
    monkeypatch.setenv("GYM_GUI_SKIP_TRAINER_DAEMON", "1")  # speed up bootstrap
    reload_settings()
    bootstrap_default_services()
    locator = get_service_locator()
    server = locator.require(JasonBridgeServer)
    assert server.is_running()
    # Host defaults to 127.0.0.1 when not overridden
    assert get_settings().jason_bridge_host == "127.0.0.1"
    assert get_settings().jason_bridge_port == 50777
