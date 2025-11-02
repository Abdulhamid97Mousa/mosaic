from __future__ import annotations

import pytest

from gym_gui.config.game_configs import FrozenLakeConfig
from gym_gui.core.adapters.toy_text import FrozenLakeV2Adapter

from spade_bdi_worker.adapters import create_adapter


@pytest.mark.parametrize(
    "goal_position",
    [
        (3, 5),
        (7, 7),
    ],
)
def test_frozenlake_v2_goal_alignment(goal_position: tuple[int, int]) -> None:
    config = FrozenLakeConfig(
        is_slippery=False,
        grid_height=8,
        grid_width=8,
        start_position=(0, 0),
        goal_position=goal_position,
        hole_count=10,
        random_holes=False,
    )

    gui_adapter = FrozenLakeV2Adapter(game_config=config)
    gui_map = gui_adapter._generate_map_descriptor()  # noqa: SLF001 - test needs generated map

    worker_adapter = create_adapter("FrozenLake-v2", game_config=config)
    worker_map = worker_adapter._generate_map_descriptor()  # type: ignore[attr-defined]  # noqa: SLF001

    assert gui_map == worker_map
    assert worker_adapter.goal_pos() == goal_position

    worker_adapter.close()
