"""Tests for MiniGrid RedBlueDoors environment integration.

Validates basic adapter wiring, action space, observation flattening, and config builder support.
"""
from __future__ import annotations

import pytest
import numpy as np

pytest.importorskip("minigrid")

from gym_gui.core.enums import GameId, ControlMode
from gym_gui.config.game_config_builder import GameConfigBuilder
from gym_gui.core.factories.adapters import create_adapter
from gym_gui.core.adapters.base import AdapterContext


@pytest.mark.parametrize("game_id", [
    GameId.MINIGRID_REDBLUE_DOORS_6x6,
    GameId.MINIGRID_REDBLUE_DOORS_8x8,
])
def test_redbluedoors_basic_load_and_observation(game_id: GameId) -> None:
    """Ensure adapter loads, reset works, and observation flattening matches expected image size + direction."""
    overrides = {}  # use defaults
    config = GameConfigBuilder.build_config(game_id, overrides)
    assert config is not None, "Config builder should return a MiniGridConfig for RedBlueDoors variants"
    assert getattr(config, "env_id", None) == game_id.value

    context = AdapterContext(settings=None, control_mode=ControlMode.AGENT_ONLY)
    adapter = create_adapter(game_id, context=context, game_config=config)
    adapter.load()
    step0 = adapter.reset(seed=123)

    obs = step0.observation
    raw = step0.info.get("_minigrid_raw_observation")
    assert isinstance(raw, dict), "Raw MiniGrid observation dict expected"
    assert "image" in raw, "Image key must be present in raw observation"
    img = np.asarray(raw["image"])
    assert img.ndim == 3, f"Expected 3D image (H,W,C), got shape {img.shape}"

    # Flatten length = image.size (+1 for direction appended)
    expected_flat_len = img.size + 1  # direction appended by adapter
    assert hasattr(obs, "size"), "Observation should be a numpy array"
    assert obs.size == expected_flat_len, f"Flattened obs length {obs.size} != expected {expected_flat_len}"

    # Action space should be Discrete(7)
    assert getattr(adapter.action_space, "n", None) == 7, "MiniGrid action space must have 7 actions"

    # Reward multiplier default
    assert getattr(config, "reward_multiplier", None) == 10.0

    # Perform a no-op / toggle sequence just to ensure step path doesn't crash
    step1 = adapter.step(6)  # 'done' action (no-op)
    assert step1 is not None
    assert hasattr(step1, "reward"), "Step result must expose reward"


if __name__ == "__main__":  # pragma: no cover
    pytest.main([__file__, "-k", "redbluedoors", "-vv"])  # Debug convenience
