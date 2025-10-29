"""Validated subprocess execution utilities with logging and security.

Provides centralized subprocess handling with:
- Type-safe command argument validation (prevents injection)
- Structured logging with log constants
- Error handling and context preservation
- Support for both sync and async subprocess operations
"""

from __future__ import annotations

import asyncio
import logging
import os
import subprocess
from typing import Optional

from pydantic import ValidationError

from gym_gui.validations import SubprocessCommand
from gym_gui.logging_config.helpers import LogConstantMixin

_LOGGER = logging.getLogger("gym_gui.subprocess_utils")


class SubprocessExecutionError(RuntimeError):
    """Raised when subprocess command validation or execution fails."""

    def __init__(self, message: str, context: Optional[dict] = None) -> None:
        """Initialize error with context.
        
        Args:
            message: Error message
            context: Optional context dict with run_id, cmd, etc.
        """
        super().__init__(message)
        self.context = context or {}


def validate_command(cmd: list[str], run_id: str = "") -> None:
    """Validate subprocess command arguments to prevent injection.
    
    Validates that all command arguments are strings using Pydantic model.
    All arguments must come from internal trusted sources (registry metadata,
    system paths, file paths).
    
    Args:
        cmd: Command argument list
        run_id: Optional run ID for logging context
        
    Raises:
        SubprocessExecutionError: If validation fails
    """
    try:
        SubprocessCommand(args=cmd)
        _LOGGER.debug(
            "Subprocess command validated",
            extra={
                "run_id": run_id,
                "cmd_length": len(cmd),
                "cmd": cmd[:3] + ["..."] if len(cmd) > 3 else cmd,
            },
        )
    except ValidationError as e:
        error_msg = f"Invalid subprocess command: {e}"
        context = {"run_id": run_id, "cmd": cmd, "error": str(e)}
        _LOGGER.error(
            error_msg,
            extra=context,
        )
        raise SubprocessExecutionError(error_msg, context=context) from e


def validated_popen(
    cmd: list[str],
    *,
    run_id: str = "",
    **popen_kwargs,
) -> subprocess.Popen:
    """Create a subprocess with validated command arguments.
    
    Validates command before passing to subprocess.Popen to prevent injection.
    
    Args:
        cmd: Command argument list
        run_id: Optional run ID for logging context
        **popen_kwargs: Additional arguments to subprocess.Popen
        
    Returns:
        subprocess.Popen instance
        
    Raises:
        SubprocessExecutionError: If command validation fails
    """
    validate_command(cmd, run_id)
    
    _LOGGER.info(
        "Spawning subprocess via Popen",
        extra={
            "run_id": run_id,
            "cmd": cmd[:2] if len(cmd) > 2 else cmd,
            "cmd_len": len(cmd),
        },
    )
    
    try:
        return subprocess.Popen(cmd, **popen_kwargs)
    except Exception as e:
        error_msg = f"Failed to spawn subprocess: {e}"
        context = {"run_id": run_id, "cmd": cmd, "error": str(e)}
        _LOGGER.error(error_msg, extra=context)
        raise SubprocessExecutionError(error_msg, context=context) from e


async def validated_create_subprocess_exec(
    *args,
    run_id: str = "",
    **exec_kwargs,
) -> asyncio.subprocess.Process:
    """Create an async subprocess with validated command arguments.
    
    Validates command before passing to asyncio.create_subprocess_exec.
    
    Args:
        *args: Command arguments (validated as list[str])
        run_id: Optional run ID for logging context
        **exec_kwargs: Additional arguments to asyncio.create_subprocess_exec
        
    Returns:
        asyncio.subprocess.Process instance
        
    Raises:
        SubprocessExecutionError: If command validation fails
    """
    cmd = list(args)
    validate_command(cmd, run_id)
    
    _LOGGER.info(
        "Spawning async subprocess",
        extra={
            "run_id": run_id,
            "cmd": cmd[:2] if len(cmd) > 2 else cmd,
            "cmd_len": len(cmd),
        },
    )
    
    try:
        return await asyncio.create_subprocess_exec(*args, **exec_kwargs)
    except Exception as e:
        error_msg = f"Failed to spawn async subprocess: {e}"
        context = {"run_id": run_id, "cmd": cmd, "error": str(e)}
        _LOGGER.error(error_msg, extra=context)
        raise SubprocessExecutionError(error_msg, context=context) from e


def validated_run(
    cmd: list[str],
    *,
    run_id: str = "",
    **run_kwargs,
) -> subprocess.CompletedProcess:
    """Run subprocess with validated command arguments.
    
    Validates command before passing to subprocess.run.
    
    Args:
        cmd: Command argument list
        run_id: Optional run ID for logging context
        **run_kwargs: Additional arguments to subprocess.run
        
    Returns:
        subprocess.CompletedProcess instance
        
    Raises:
        SubprocessExecutionError: If command validation fails
    """
    validate_command(cmd, run_id)
    
    _LOGGER.info(
        "Running subprocess",
        extra={
            "run_id": run_id,
            "cmd": cmd[:2] if len(cmd) > 2 else cmd,
            "cmd_len": len(cmd),
        },
    )
    
    try:
        return subprocess.run(cmd, **run_kwargs)
    except Exception as e:
        error_msg = f"Failed to run subprocess: {e}"
        context = {"run_id": run_id, "cmd": cmd, "error": str(e)}
        _LOGGER.error(error_msg, extra=context)
        raise SubprocessExecutionError(error_msg, context=context) from e


__all__ = [
    "SubprocessExecutionError",
    "validate_command",
    "validated_popen",
    "validated_create_subprocess_exec",
    "validated_run",
]
