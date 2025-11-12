"""Integration tests for MiniGrid RedBlueDoors environment variants.

Verifies that MiniGrid-RedBlueDoors-6x6-v0 and MiniGrid-RedBlueDoors-8x8-v0
are fully wired into the GUI adapter stack:
- GameId enum membership
- Default config export
- MINIGRID_GAME_IDS tuple membership
- Documentation retrieval via get_game_info
- Adapter load/reset/step behaviour
- Observation flattening (image flatten + direction byte)
- Logging codes emitted on boot
"""
from __future__ import annotations

import logging
import numpy as np
import pytest

pytest.importorskip("minigrid")

from gym_gui.core.adapters.base import AdapterContext
from gym_gui.core.adapters.minigrid import MiniGridAdapter
from gym_gui.config.game_configs import (
    DEFAULT_MINIGRID_REDBLUE_DOORS_6x6_CONFIG,
    DEFAULT_MINIGRID_REDBLUE_DOORS_8x8_CONFIG,
    MiniGridConfig,
)
from gym_gui.core.enums import ControlMode, GameId
from gym_gui.game_docs import get_game_info
from gym_gui.logging_config.log_constants import LOG_ENV_MINIGRID_BOOT
from gym_gui.ui.environments.single_agent_env.minigrid.config_panel import (
    MINIGRID_GAME_IDS,
    resolve_default_config,
)

# Parametrized variants (GameId, default config, label)
REDBLUE_VARIANTS = [
    (GameId.MINIGRID_REDBLUE_DOORS_6x6, DEFAULT_MINIGRID_REDBLUE_DOORS_6x6_CONFIG, "6x6"),
    (GameId.MINIGRID_REDBLUE_DOORS_8x8, DEFAULT_MINIGRID_REDBLUE_DOORS_8x8_CONFIG, "8x8"),
]

class TestRedBlueDoorsRegistration:
    def test_game_ids_exist(self) -> None:
        expected = [v[0] for v in REDBLUE_VARIANTS]
        for gid in expected:
            assert gid in GameId
            assert "RedBlueDoors" in gid.value
            assert gid.value.startswith("MiniGrid-RedBlueDoors-")

    def test_default_configs_exported(self) -> None:
        from gym_gui.config import game_configs
        for name in [
            "DEFAULT_MINIGRID_REDBLUE_DOORS_6x6_CONFIG",
            "DEFAULT_MINIGRID_REDBLUE_DOORS_8x8_CONFIG",
        ]:
            assert hasattr(game_configs, name)
            cfg = getattr(game_configs, name)
            assert isinstance(cfg, MiniGridConfig)
            assert "RedBlueDoors" in cfg.env_id

    def test_membership_in_minigrid_game_ids(self) -> None:
        for gid, _cfg, _label in REDBLUE_VARIANTS:
            assert gid in MINIGRID_GAME_IDS

    def test_resolve_default_config_matches(self) -> None:
        for gid, cfg, _label in REDBLUE_VARIANTS:
            resolved = resolve_default_config(gid)
            assert resolved.env_id == cfg.env_id == gid.value

class TestRedBlueDoorsDocs:
    @pytest.mark.parametrize("gid,cfg,label", REDBLUE_VARIANTS)
    def test_get_game_info_contains_keywords(self, gid: GameId, cfg: MiniGridConfig, label: str) -> None:
        html = get_game_info(gid)
        assert html and len(html) > 0
        assert "RedBlueDoors" in html
        assert "MiniGrid" in html
        assert label.replace("x", "×")[:1] in html or label in html  # size reference present (best effort)

class TestRedBlueDoorsAdapter:
    @pytest.mark.parametrize("gid,cfg,label", REDBLUE_VARIANTS)
    def test_adapter_load_and_reset(self, gid: GameId, cfg: MiniGridConfig, label: str) -> None:
        context = AdapterContext(settings=None, control_mode=ControlMode.AGENT_ONLY)
        adapter = MiniGridAdapter(context, config=cfg)
        try:
            adapter.load()
            assert adapter._env is not None
            step = adapter.reset(seed=123)
            assert step.state.environment["env_id"] == gid.value
            assert step.observation.ndim == 1
            assert step.observation.dtype == np.uint8
        finally:
            adapter.close()

    @pytest.mark.parametrize("gid,cfg,label", REDBLUE_VARIANTS)
    def test_observation_flattening(self, gid: GameId, cfg: MiniGridConfig, label: str) -> None:
        context = AdapterContext(settings=None, control_mode=ControlMode.AGENT_ONLY)
        adapter = MiniGridAdapter(context, config=cfg)
        try:
            adapter.load()
            _ = adapter.reset(seed=99)
            step = adapter.step(2)  # move forward action
            raw = step.info.get("_minigrid_raw_observation")
            assert isinstance(raw, dict)
            image = raw.get("image")
            assert isinstance(image, np.ndarray)
            h, w, c = image.shape
            expected_flat_len = h * w * c + 1  # + direction byte
            assert step.observation.shape[0] == expected_flat_len
        finally:
            adapter.close()

    @pytest.mark.parametrize("gid,cfg,label", REDBLUE_VARIANTS)
    def test_action_space_discrete_7(self, gid: GameId, cfg: MiniGridConfig, label: str) -> None:
        context = AdapterContext(settings=None, control_mode=ControlMode.AGENT_ONLY)
        adapter = MiniGridAdapter(context, config=cfg)
        try:
            adapter.load()
            action_space = adapter.action_space
            assert hasattr(action_space, "n")
            assert int(action_space.n) == 7  # type: ignore[attr-defined]
        finally:
            adapter.close()

class TestRedBlueDoorsLogging:
    @pytest.mark.parametrize("gid,cfg,label", REDBLUE_VARIANTS)
    def test_boot_log_emitted(self, gid: GameId, cfg: MiniGridConfig, label: str, caplog: pytest.LogCaptureFixture) -> None:
        caplog.set_level(logging.INFO, logger="gym_gui.core.adapters.base")
        context = AdapterContext(settings=None, control_mode=ControlMode.AGENT_ONLY)
        adapter = MiniGridAdapter(context, config=cfg)
        try:
            adapter.load()
            _ = adapter.reset(seed=1)
            boot_codes = [getattr(r, "log_code", None) for r in caplog.records]
            boot_codes = [c for c in boot_codes if c is not None]
            assert LOG_ENV_MINIGRID_BOOT.code in boot_codes
        finally:
            adapter.close()

# Summary test

def test_redbluedoors_integration_summary() -> None:
    context = AdapterContext(settings=None, control_mode=ControlMode.AGENT_ONLY)
    for gid, cfg, label in REDBLUE_VARIANTS:
        adapter = MiniGridAdapter(context, config=cfg)
        try:
            adapter.load()
            step = adapter.reset(seed=42)
            assert step is not None
            print(f"\u2713 RedBlueDoors {label}: {gid.value}")
        finally:
            adapter.close()
    print(f"\n\u2713 All {len(REDBLUE_VARIANTS)} RedBlueDoors variants integrated successfully!")
