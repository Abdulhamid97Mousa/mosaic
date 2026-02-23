"""
Algorithm-agnostic checkpoint saving and loading utilities.

This module provides utilities for saving and loading model checkpoints
that work across different RL algorithms (PPO, A2C, DQN, etc.).

Usage:
    from cleanrl_worker.save import save_checkpoint, load_checkpoint

    # Save checkpoint during training
    save_checkpoint(
        path="checkpoints/model.pt",
        agent=agent,
        optimizer=optimizer,
        global_step=10000,
        args=training_args,
    )

    # Load checkpoint for resume or evaluation
    checkpoint = load_checkpoint("checkpoints/model.pt", agent, optimizer)
"""

import logging
from pathlib import Path
from typing import Any, Dict, Optional, Union

import torch
import torch.nn as nn
import torch.optim as optim

logger = logging.getLogger(__name__)


def save_checkpoint(
    path: Union[str, Path],
    agent: nn.Module,
    optimizer: Optional[optim.Optimizer] = None,
    global_step: int = 0,
    iteration: int = 0,
    args: Optional[Any] = None,
    extra: Optional[Dict[str, Any]] = None,
) -> str:
    """Save a training checkpoint.

    Saves the agent's state dict along with training state for proper resume.

    Args:
        path: Path to save the checkpoint file
        agent: The neural network agent to save
        optimizer: Optional optimizer to save for resume
        global_step: Current global step count
        iteration: Current iteration/epoch number
        args: Optional training arguments/config to save
        extra: Optional dictionary of extra data to save

    Returns:
        The path where the checkpoint was saved

    Example:
        >>> save_checkpoint(
        ...     "runs/my_run/checkpoint.pt",
        ...     agent=my_agent,
        ...     optimizer=adam_optimizer,
        ...     global_step=50000,
        ...     args=training_args,
        ... )
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    checkpoint = {
        "model_state_dict": agent.state_dict(),
        "global_step": global_step,
        "iteration": iteration,
    }

    if optimizer is not None:
        checkpoint["optimizer_state_dict"] = optimizer.state_dict()

    if args is not None:
        # Convert dataclass to dict if needed
        if hasattr(args, "__dataclass_fields__"):
            import dataclasses
            checkpoint["args"] = dataclasses.asdict(args)
        else:
            checkpoint["args"] = vars(args) if hasattr(args, "__dict__") else args

    if extra is not None:
        checkpoint["extra"] = extra

    torch.save(checkpoint, path)
    logger.info("Checkpoint saved to %s (global_step=%d)", path, global_step)

    return str(path)


def load_checkpoint(
    path: Union[str, Path],
    agent: Optional[nn.Module] = None,
    optimizer: Optional[optim.Optimizer] = None,
    device: Optional[torch.device] = None,
    strict: bool = True,
) -> Dict[str, Any]:
    """Load a training checkpoint.

    Loads the agent's state dict and optionally the optimizer state.

    Args:
        path: Path to the checkpoint file
        agent: Optional agent to load weights into
        optimizer: Optional optimizer to load state into
        device: Device to map tensors to (default: auto-detect)
        strict: Whether to strictly enforce matching state dict keys

    Returns:
        Dictionary containing checkpoint data including:
        - global_step: The step count when checkpoint was saved
        - iteration: The iteration/epoch when saved
        - args: Training arguments if saved
        - extra: Extra data if saved

    Example:
        >>> checkpoint = load_checkpoint(
        ...     "runs/my_run/checkpoint.pt",
        ...     agent=my_agent,
        ...     optimizer=adam_optimizer,
        ... )
        >>> start_step = checkpoint["global_step"]
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    checkpoint = torch.load(path, map_location=device)

    if agent is not None and "model_state_dict" in checkpoint:
        agent.load_state_dict(checkpoint["model_state_dict"], strict=strict)
        logger.info("Loaded agent state from %s", path)

    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        logger.info("Loaded optimizer state from %s", path)

    global_step = checkpoint.get("global_step", 0)
    logger.info("Checkpoint loaded from %s (global_step=%d)", path, global_step)

    return checkpoint


def get_latest_checkpoint(checkpoint_dir: Union[str, Path], pattern: str = "*.pt") -> Optional[Path]:
    """Find the latest checkpoint in a directory.

    Searches for checkpoint files matching the pattern and returns
    the most recently modified one.

    Args:
        checkpoint_dir: Directory to search for checkpoints
        pattern: Glob pattern for checkpoint files

    Returns:
        Path to the latest checkpoint, or None if no checkpoints found

    Example:
        >>> latest = get_latest_checkpoint("runs/my_run/checkpoints/")
        >>> if latest:
        ...     checkpoint = load_checkpoint(latest, agent)
    """
    checkpoint_dir = Path(checkpoint_dir)
    if not checkpoint_dir.exists():
        return None

    checkpoints = list(checkpoint_dir.glob(pattern))
    if not checkpoints:
        return None

    # Sort by modification time, newest first
    checkpoints.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return checkpoints[0]


def save_best_checkpoint(
    path: Union[str, Path],
    agent: nn.Module,
    current_metric: float,
    best_metric: float,
    higher_is_better: bool = True,
    optimizer: Optional[optim.Optimizer] = None,
    global_step: int = 0,
    **kwargs,
) -> tuple[str, float]:
    """Save checkpoint only if the current metric is the best so far.

    Args:
        path: Path to save the best checkpoint
        agent: The neural network agent to save
        current_metric: The current evaluation metric value
        best_metric: The previous best metric value
        higher_is_better: If True, higher metric is better (e.g., reward)
                         If False, lower is better (e.g., loss)
        optimizer: Optional optimizer to save
        global_step: Current global step count
        **kwargs: Additional arguments passed to save_checkpoint

    Returns:
        Tuple of (saved_path or empty string, new best metric)

    Example:
        >>> best_return = float("-inf")
        >>> for epoch in range(epochs):
        ...     current_return = evaluate(agent)
        ...     _, best_return = save_best_checkpoint(
        ...         "checkpoints/best.pt",
        ...         agent=agent,
        ...         current_metric=current_return,
        ...         best_metric=best_return,
        ...         higher_is_better=True,
        ...     )
    """
    is_better = (
        current_metric > best_metric if higher_is_better else current_metric < best_metric
    )

    if is_better:
        saved_path = save_checkpoint(
            path=path,
            agent=agent,
            optimizer=optimizer,
            global_step=global_step,
            extra={"metric": current_metric},
            **kwargs,
        )
        logger.info(
            "New best checkpoint! metric=%.4f (previous=%.4f)",
            current_metric,
            best_metric,
        )
        return saved_path, current_metric

    return "", best_metric
