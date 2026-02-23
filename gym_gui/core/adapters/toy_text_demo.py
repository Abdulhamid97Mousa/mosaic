"""CLI smoke harness for toy-text adapters."""

from __future__ import annotations

import argparse
import logging
from typing import Sequence

from gym_gui.config.settings import get_settings
from gym_gui.core.adapters.base import AdapterContext
from gym_gui.core.adapters.toy_text import TOY_TEXT_ADAPTERS
from gym_gui.core.enums import ControlMode, GameId
from gym_gui.logging_config.logger import configure_logging


def _choices() -> Sequence[str]:
    return [game_id.value for game_id in TOY_TEXT_ADAPTERS]


def main() -> int:
    parser = argparse.ArgumentParser(description="Run toy-text adapters without the Qt GUI")
    parser.add_argument("--env", choices=_choices(), default=GameId.FROZEN_LAKE.value)
    parser.add_argument("--steps", type=int, default=8, help="Number of random actions to execute")
    parser.add_argument("--seed", type=int, default=0, help="Seed for environment reset")
    args = parser.parse_args()

    configure_logging(level=logging.DEBUG)
    settings = get_settings()
    control_mode = settings.default_control_mode or ControlMode.HUMAN_ONLY
    context = AdapterContext(
        settings=settings,
        control_mode=control_mode,
        logger_factory=logging.getLogger,
    )

    game_id = GameId(args.env)
    adapter_cls = TOY_TEXT_ADAPTERS[game_id]
    adapter = adapter_cls(context)
    adapter.ensure_control_mode(control_mode)
    adapter.load()
    state = adapter.reset(seed=args.seed)

    print(f"Initial state observation={state.observation}")
    _print_grid(state.render_payload)

    for step_index in range(1, args.steps + 1):
        action = adapter.action_space.sample()
        state = adapter.step(action)
        print(
            f"step={step_index} action={action} reward={state.reward:.2f} "
            f"terminated={state.terminated} truncated={state.truncated}"
        )
        _print_grid(state.render_payload)
        if state.terminated or state.truncated:
            break

    adapter.close()
    return 0


def _print_grid(render_payload: dict[str, object] | None) -> None:
    if not render_payload or "grid" not in render_payload:
        print("[render] payload unavailable")
        return
    grid_obj = render_payload["grid"]
    if not isinstance(grid_obj, list):
        print(f"[render] unexpected grid payload: {type(grid_obj)!r}")
        return
    grid: list[list[str]] = []
    for row in grid_obj:
        if isinstance(row, list):
            grid.append([str(cell) for cell in row])
        else:
            grid.append([str(row)])
    print("[render] grid view:")
    for row in grid:
        print("".join(row))
    print()


if __name__ == "__main__":
    raise SystemExit(main())
