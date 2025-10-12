"""Headless keyboard orchestrator for quick adapter experiments."""

from __future__ import annotations

import argparse
import logging
from typing import Iterable, Tuple

import gymnasium as gym

from gym_gui.config.settings import get_settings
from gym_gui.core.adapters.base import AdapterContext, AdapterStep
from gym_gui.core.factories import available_games, create_adapter
from gym_gui.core.enums import ControlMode, GameId
from gym_gui.logging_config.logger import configure_logging


def _game_choices() -> Iterable[str]:
    return [game.value for game in available_games()]


def _coerce_control_mode(value: str) -> ControlMode:
    try:
        return ControlMode(value)
    except ValueError as exc:  # pragma: no cover - argparse validation handles this
        raise argparse.ArgumentTypeError(str(exc)) from exc


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Manual keyboard orchestrator for Gym GUI adapters")
    parser.add_argument("--env", choices=_game_choices(), default=GameId.FROZEN_LAKE.value)
    parser.add_argument(
        "--mode",
        type=_coerce_control_mode,
        default=ControlMode.HUMAN_ONLY,
        help="Control mode to validate (must be supported by the adapter)",
    )
    parser.add_argument("--seed", type=int, default=1, help="Seed passed to adapter.reset")
    parser.add_argument("--max-steps", type=int, default=100, help="Maximum number of actions before exiting")
    parser.add_argument("--log-level", default="DEBUG", help="Logging level (e.g. INFO, DEBUG)")
    args = parser.parse_args(argv)

    log_level = getattr(logging, args.log_level.upper(), logging.DEBUG)
    configure_logging(level=log_level, stream=True, log_to_file=True)
    logger = logging.getLogger("gym_gui.controllers.cli")

    settings = get_settings()
    context = AdapterContext(
        settings=settings,
        control_mode=args.mode,
        logger_factory=logging.getLogger,
    )

    game_id = GameId(args.env)
    adapter = create_adapter(game_id, context)
    adapter.ensure_control_mode(args.mode)
    adapter.load()
    step = adapter.reset(seed=args.seed)
    _print_step(step, step_index=0)

    try:
        step_index = 0
        while step_index < args.max_steps:
            command, action = _prompt_action(adapter.action_space)
            if command == "quit":
                logger.info("Exiting orchestrator at user request")
                break
            if command == "reset":
                step_index = 0
                step = adapter.reset(seed=args.seed)
                _print_step(step, step_index)
                continue

            assert action is not None
            step_index += 1
            step = adapter.step(action)
            _print_step(step, step_index)
            if step.terminated or step.truncated:
                logger.info("Episode finished at step=%s", step_index)
                break
    finally:
        adapter.close()
    return 0


def _prompt_action(space: gym.Space) -> Tuple[str, int | None]:
    if not isinstance(space, gym.spaces.Discrete):
        raise NotImplementedError("CLI orchestrator only supports discrete action spaces so far")

    max_action = space.n - 1
    while True:
        raw = input(f"Enter action [0-{max_action}] (q to quit, r to reset): ").strip()
        if raw.lower() in {"q", "quit", "exit"}:
            return "quit", None
        if raw.lower() == "r":
            return "reset", None
        if raw.isdigit():
            value = int(raw)
            if 0 <= value <= max_action:
                return "action", value
        print("Invalid input. Please enter a number in range or 'q' to quit.")


def _print_step(step: AdapterStep, step_index: int) -> None:
    render_payload = step.render_payload
    print(
        f"step={step_index} reward={step.reward:.2f} terminated={step.terminated} "
        f"truncated={step.truncated}"
    )
    if isinstance(render_payload, dict) and "grid" in render_payload:
        _print_grid(render_payload["grid"])


def _print_grid(grid: object) -> None:
    if not isinstance(grid, list):
        print("[render] payload unavailable")
        return
    print("[render] grid view:")
    for row in grid:
        if isinstance(row, list):
            print("".join(str(cell) for cell in row))
        else:
            print(str(row))
    print()


if __name__ == "__main__":  # pragma: no cover - CLI convenience
    raise SystemExit(main())
