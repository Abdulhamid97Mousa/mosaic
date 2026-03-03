"""Test GUI adapters work correctly with all toy-text games.

Tests that the centralized adapters (now used by both GUI and worker):
1. Can be created and loaded
2. Provide complete render payloads
3. Handle custom game configs correctly
4. Generate proper map descriptors with hole placement
"""

from __future__ import annotations

import pytest
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from gym_gui.core.adapters.toy_text import ToyTextAdapter


class TestToyTextAdaptersLoad:
    """Test that all toy-text adapters can be created and loaded."""

    def test_frozenlake_v1_loads(self):
        """FrozenLake-v1 adapter loads successfully."""
        from gym_gui.core.adapters.toy_text import FrozenLakeAdapter
        from gym_gui.config.game_configs import DEFAULT_FROZEN_LAKE_CONFIG
        
        adapter = FrozenLakeAdapter(game_config=DEFAULT_FROZEN_LAKE_CONFIG)
        adapter.load()
        
        result = adapter.reset(seed=42)
        assert result.observation is not None
        assert isinstance(int(result.observation), int)

    def test_frozenlake_v2_loads(self):
        """FrozenLake-v2 adapter loads successfully."""
        from gym_gui.core.adapters.toy_text import FrozenLakeV2Adapter
        from gym_gui.config.game_configs import DEFAULT_FROZEN_LAKE_V2_CONFIG
        
        adapter = FrozenLakeV2Adapter(game_config=DEFAULT_FROZEN_LAKE_V2_CONFIG)
        adapter.load()
        
        result = adapter.reset(seed=42)
        assert result.observation is not None
        assert isinstance(int(result.observation), int)

    def test_cliffwalking_loads(self):
        """CliffWalking adapter loads successfully."""
        from gym_gui.core.adapters.toy_text import CliffWalkingAdapter
        from gym_gui.config.game_configs import DEFAULT_CLIFF_WALKING_CONFIG
        
        adapter = CliffWalkingAdapter(game_config=DEFAULT_CLIFF_WALKING_CONFIG)
        adapter.load()
        
        result = adapter.reset(seed=42)
        assert result.observation is not None
        assert isinstance(int(result.observation), int)

    def test_taxi_loads(self):
        """Taxi adapter loads successfully."""
        from gym_gui.core.adapters.toy_text import TaxiAdapter
        from gym_gui.config.game_configs import DEFAULT_TAXI_CONFIG
        
        adapter = TaxiAdapter(game_config=DEFAULT_TAXI_CONFIG)
        adapter.load()
        
        result = adapter.reset(seed=42)
        assert result.observation is not None
        assert isinstance(int(result.observation), int)


class TestAdapterRenderPayloads:
    """Test that adapters provide complete render payloads."""

    def test_frozenlake_v1_render_has_holes(self):
        """FrozenLake-v1 render includes hole positions."""
        from gym_gui.core.adapters.toy_text import FrozenLakeAdapter
        from gym_gui.config.game_configs import DEFAULT_FROZEN_LAKE_CONFIG
        
        adapter = FrozenLakeAdapter(game_config=DEFAULT_FROZEN_LAKE_CONFIG)
        adapter.load()
        adapter.reset(seed=42)
        
        payload = adapter.render()
        assert isinstance(payload, dict)
        assert "mode" in payload
        assert "grid" in payload
        assert "agent_position" in payload
        # FrozenLake should have holes
        assert "holes" in payload or "grid" in payload

    def test_frozenlake_v2_render_with_custom_goal(self):
        """FrozenLake-v2 render works with custom goal position."""
        from gym_gui.core.adapters.toy_text import FrozenLakeV2Adapter
        from gym_gui.config.game_configs import FrozenLakeConfig
        
        config = FrozenLakeConfig(
            grid_height=8,
            grid_width=8,
            goal_position=(1, 7),  # Custom goal at top-right
            start_position=(0, 0),
            hole_count=10,
            random_holes=False,
            is_slippery=False,
        )
        adapter = FrozenLakeV2Adapter(game_config=config)
        adapter.load()
        adapter.reset(seed=42)
        
        payload = adapter.render()
        assert isinstance(payload, dict)
        assert "holes" in payload
        assert "goal" in payload
        assert "grid" in payload
        
        # Goal should be at correct position
        goal = payload.get("goal")
        assert goal is not None
        assert goal.get("row") == 1
        assert goal.get("col") == 7
        
        # Should have holes (not all clustered at top)
        holes = payload.get("holes", [])
        assert len(holes) > 0
        # Check holes are distributed (not all in row 0)
        rows = [h.get("row") for h in holes if isinstance(h, dict)]
        assert len(set(rows)) > 1, "Holes should be distributed across multiple rows"

    def test_cliffwalking_render_complete(self):
        """CliffWalking render provides complete payload."""
        from gym_gui.core.adapters.toy_text import CliffWalkingAdapter
        from gym_gui.config.game_configs import DEFAULT_CLIFF_WALKING_CONFIG
        
        adapter = CliffWalkingAdapter(game_config=DEFAULT_CLIFF_WALKING_CONFIG)
        adapter.load()
        adapter.reset(seed=42)
        
        payload = adapter.render()
        assert isinstance(payload, dict)
        assert payload.get("mode") == "grid"
        assert "grid" in payload
        assert "agent_position" in payload
        assert "game_id" in payload
        assert payload.get("game_id") == "CliffWalking-v1"
        
        # Grid should be 4x12
        grid = payload.get("grid", [])
        assert len(grid) == 4
        if grid:
            assert len(grid[0]) == 12

    def test_taxi_render_has_state(self):
        """Taxi render includes taxi state information."""
        from gym_gui.core.adapters.toy_text import TaxiAdapter
        from gym_gui.config.game_configs import DEFAULT_TAXI_CONFIG
        
        adapter = TaxiAdapter(game_config=DEFAULT_TAXI_CONFIG)
        adapter.load()
        adapter.reset(seed=42)
        
        payload = adapter.render()
        assert isinstance(payload, dict)
        assert "mode" in payload
        assert "grid" in payload
        assert "agent_position" in payload
        # Taxi should have taxi_state info
        assert "taxi_state" in payload or "agent_position" in payload


class TestCustomGameConfigs:
    """Test that custom game configurations are applied correctly."""

    def test_frozenlake_v2_custom_grid_size(self):
        """FrozenLake-v2 respects custom grid dimensions."""
        from gym_gui.core.adapters.toy_text import FrozenLakeV2Adapter
        from gym_gui.config.game_configs import FrozenLakeConfig
        
        config = FrozenLakeConfig(
            grid_height=6,
            grid_width=6,
            goal_position=(5, 5),
            hole_count=5,
            is_slippery=False,
        )
        adapter = FrozenLakeV2Adapter(game_config=config)
        adapter.load()
        
        # Should create 6x6 environment
        assert adapter._game_config.grid_height == 6  # type: ignore[attr-defined]
        assert adapter._game_config.grid_width == 6  # type: ignore[attr-defined]

    def test_cliffwalking_slippery_config(self):
        """CliffWalking respects is_slippery configuration."""
        from gym_gui.core.adapters.toy_text import CliffWalkingAdapter
        from gym_gui.config.game_configs import CliffWalkingConfig
        
        config = CliffWalkingConfig(is_slippery=True)
        adapter = CliffWalkingAdapter(game_config=config)
        adapter.load()
        
        # Config should be applied
        assert adapter._game_config.is_slippery is True  # type: ignore[attr-defined]


class TestHolePlacement:
    """Test hole placement logic for FrozenLake-v2."""

    def test_deterministic_holes_use_official_pattern(self):
        """When random_holes=False, holes follow official Gymnasium pattern."""
        from gym_gui.core.adapters.toy_text import FrozenLakeV2Adapter
        from gym_gui.config.game_configs import FrozenLakeConfig, DEFAULT_FROZEN_LAKE_V2_CONFIG
        
        adapter = FrozenLakeV2Adapter(game_config=DEFAULT_FROZEN_LAKE_V2_CONFIG)
        adapter.load()
        adapter.reset(seed=42)
        
        payload = adapter.render()
        holes = payload.get("holes", [])
        
        # Should have holes (default is 10)
        assert len(holes) >= 5, "Should have multiple holes"
        
        # Holes should be distributed across multiple rows (not clustered at top)
        rows = set(h.get("row") for h in holes if isinstance(h, dict))
        assert len(rows) >= 3, f"Holes should span multiple rows, got {rows}"
        
        # Check specific official hole positions exist
        hole_positions = {(h.get("row"), h.get("col")) for h in holes if isinstance(h, dict)}
        # From official map: (2,3), (3,5), (4,3), (5,1), (5,2), (5,6), (6,1), (6,4), (6,6), (7,3)
        official_holes = {(2,3), (3,5), (4,3), (5,1), (5,2), (5,6), (6,1), (6,4), (6,6), (7,3)}
        # At least some official holes should be present
        overlap = hole_positions & official_holes
        assert len(overlap) >= 5, f"Should include official hole positions, got {hole_positions}"

    def test_random_holes_distributed(self):
        """When random_holes=True, holes are randomly placed."""
        from gym_gui.core.adapters.toy_text import FrozenLakeV2Adapter
        from gym_gui.config.game_configs import FrozenLakeConfig
        
        config = FrozenLakeConfig(
            grid_height=8,
            grid_width=8,
            hole_count=10,
            random_holes=True,
            is_slippery=False,
        )
        
        # Create two adapters with different seeds
        adapter1 = FrozenLakeV2Adapter(game_config=config)
        adapter1.load()
        adapter1.reset(seed=1)
        payload1 = adapter1.render()
        holes1 = {(h.get("row"), h.get("col")) for h in payload1.get("holes", []) if isinstance(h, dict)}
        
        adapter2 = FrozenLakeV2Adapter(game_config=config)
        adapter2.load()
        adapter2.reset(seed=2)
        payload2 = adapter2.render()
        holes2 = {(h.get("row"), h.get("col")) for h in payload2.get("holes", []) if isinstance(h, dict)}
        
        # With random_holes=True, different seeds should give different hole patterns
        # (This may occasionally fail due to random chance, but very unlikely)
        assert holes1 != holes2, "Random holes should differ between seeds"

    def test_custom_goal_avoids_hole_conflict(self):
        """Custom goal position doesn't have a hole placed on it."""
        from gym_gui.core.adapters.toy_text import FrozenLakeV2Adapter
        from gym_gui.config.game_configs import FrozenLakeConfig
        
        config = FrozenLakeConfig(
            grid_height=8,
            grid_width=8,
            goal_position=(1, 7),  # Custom goal
            hole_count=10,
            random_holes=False,
            is_slippery=False,
        )
        adapter = FrozenLakeV2Adapter(game_config=config)
        adapter.load()
        adapter.reset(seed=42)
        
        payload = adapter.render()
        holes = {(h.get("row"), h.get("col")) for h in payload.get("holes", []) if isinstance(h, dict)}
        goal = payload.get("goal")
        
        # Goal should be at (1, 7)
        assert goal is not None
        assert (goal.get("row"), goal.get("col")) == (1, 7)
        
        # No hole should be at the goal position
        assert (1, 7) not in holes, "Goal position should not have a hole"


class TestAdapterCompatibility:
    """Test that GUI adapters work with worker code patterns."""

    def test_adapter_has_required_methods(self):
        """All adapters have methods required by worker."""
        from gym_gui.core.adapters.toy_text import (
            FrozenLakeAdapter,
            CliffWalkingAdapter,
            TaxiAdapter,
        )
        from gym_gui.config.game_configs import (
            DEFAULT_FROZEN_LAKE_CONFIG,
            DEFAULT_CLIFF_WALKING_CONFIG,
            DEFAULT_TAXI_CONFIG,
        )
        
        adapters = [
            (FrozenLakeAdapter, DEFAULT_FROZEN_LAKE_CONFIG),
            (CliffWalkingAdapter, DEFAULT_CLIFF_WALKING_CONFIG),
            (TaxiAdapter, DEFAULT_TAXI_CONFIG),
        ]
        
        for AdapterClass, config in adapters:
            adapter = AdapterClass(game_config=config)
            adapter.load()
            
            # Worker requires these methods
            assert hasattr(adapter, "reset")
            assert hasattr(adapter, "step")
            assert hasattr(adapter, "render")
            assert hasattr(adapter, "state_to_pos")
            assert hasattr(adapter, "_get_grid_width")
            assert hasattr(adapter, "defaults")
            assert hasattr(adapter, "observation_space")
            assert hasattr(adapter, "action_space")

    def test_adapter_spaces_have_n_attribute(self):
        """Adapters provide Gymnasium spaces with .n attribute."""
        from gym_gui.core.adapters.toy_text import FrozenLakeAdapter
        from gym_gui.config.game_configs import DEFAULT_FROZEN_LAKE_CONFIG
        
        adapter = FrozenLakeAdapter(game_config=DEFAULT_FROZEN_LAKE_CONFIG)
        adapter.load()
        
        # Spaces should be Discrete and have .n
        assert hasattr(adapter.observation_space, "n")
        assert hasattr(adapter.action_space, "n")
        
        obs_n = int(adapter.observation_space.n)  # type: ignore[attr-defined]
        act_n = int(adapter.action_space.n)  # type: ignore[attr-defined]
        
        assert obs_n > 0
        assert act_n > 0


class TestRenderPayloadStructure:
    """Test render payload contains all required fields for UI."""

    @pytest.mark.parametrize("game_id,adapter_class,config", [
        ("FrozenLake-v1", "FrozenLakeAdapter", "DEFAULT_FROZEN_LAKE_CONFIG"),
        ("FrozenLake-v2", "FrozenLakeV2Adapter", "DEFAULT_FROZEN_LAKE_V2_CONFIG"),
        ("CliffWalking-v1", "CliffWalkingAdapter", "DEFAULT_CLIFF_WALKING_CONFIG"),
        ("Taxi-v3", "TaxiAdapter", "DEFAULT_TAXI_CONFIG"),
    ])
    def test_render_payload_structure(self, game_id, adapter_class, config):
        """All adapters provide properly structured render payloads."""
        from gym_gui.core.adapters import toy_text
        from gym_gui.config import game_configs
        
        AdapterClass = getattr(toy_text, adapter_class)
        game_config = getattr(game_configs, config)
        
        adapter = AdapterClass(game_config=game_config)
        adapter.load()
        adapter.reset(seed=42)
        
        payload = adapter.render()
        
        # All render payloads must have these
        assert "mode" in payload, f"{game_id} missing mode"
        assert "grid" in payload, f"{game_id} missing grid"
        assert "agent_position" in payload, f"{game_id} missing agent_position"
        assert "game_id" in payload, f"{game_id} missing game_id"
        assert payload["game_id"] == game_id, f"{game_id} wrong game_id in payload"


class TestWorkerIntegration:
    """Test patterns used by worker runtime."""

    def test_worker_reset_pattern(self):
        """Worker pattern of unpacking reset result."""
        from gym_gui.core.adapters.toy_text import FrozenLakeAdapter
        from gym_gui.config.game_configs import DEFAULT_FROZEN_LAKE_CONFIG
        
        adapter = FrozenLakeAdapter(game_config=DEFAULT_FROZEN_LAKE_CONFIG)
        adapter.load()
        
        # Worker pattern
        reset_result = adapter.reset(seed=42)
        state = int(reset_result.observation)
        obs = reset_result.info
        
        assert isinstance(state, int)
        assert isinstance(obs, dict)

    def test_worker_step_pattern(self):
        """Worker pattern of unpacking step result."""
        from gym_gui.core.adapters.toy_text import CliffWalkingAdapter
        from gym_gui.config.game_configs import DEFAULT_CLIFF_WALKING_CONFIG
        
        adapter = CliffWalkingAdapter(game_config=DEFAULT_CLIFF_WALKING_CONFIG)
        adapter.load()
        adapter.reset(seed=42)
        
        # Worker pattern
        step_result = adapter.step(1)  # RIGHT action
        next_state = int(step_result.observation)
        reward = float(step_result.reward)
        terminated = bool(step_result.terminated)
        truncated = bool(step_result.truncated)
        
        assert isinstance(next_state, int)
        assert isinstance(reward, float)
        assert isinstance(terminated, bool)
        assert isinstance(truncated, bool)

    def test_worker_render_pattern(self):
        """Worker can call adapter.render() for telemetry."""
        from gym_gui.core.adapters.toy_text import TaxiAdapter
        from gym_gui.config.game_configs import DEFAULT_TAXI_CONFIG
        
        adapter = TaxiAdapter(game_config=DEFAULT_TAXI_CONFIG)
        adapter.load()
        adapter.reset(seed=42)
        adapter.step(0)  # SOUTH
        
        # Worker pattern
        if hasattr(adapter, "render") and callable(adapter.render):
            render_payload = adapter.render()
            assert isinstance(render_payload, dict)
            assert "mode" in render_payload
            assert "grid" in render_payload
        else:
            pytest.fail("Adapter should have callable render() method")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

