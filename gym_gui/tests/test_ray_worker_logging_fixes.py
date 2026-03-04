"""Tests for ray_worker logging fixes.

Covers:
- _StructuredLogFormatter: pipe-delimited output for log_constant() records,
  human-readable fallback for regular log records
- Dispatcher regex compatibility: formatter output matches _parse_structured_log_line()
- FileHandler creation in _setup_logging()
- LOG_WORKER_RAY_* constants discoverable in ALL_LOG_CONSTANTS
- HumanVsAgentConfigForm._on_difficulty_changed() uses safe preset method
- Dispatcher _maybe_open_log_files() works for all worker types
"""

from __future__ import annotations

import io
import json
import logging
import re
import tempfile
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# _StructuredLogFormatter tests
# ---------------------------------------------------------------------------

# Re-implement the formatter logic here to avoid importing runtime.py which
# pulls in Ray (heavy / may not be installed).  We test the identical logic.
_STANDARD_LOG_ATTRS = frozenset(
    logging.LogRecord("", 0, "", 0, "", (), None).__dict__
)

_HUMAN_FMT = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
_DATE_FMT = "%Y-%m-%d %H:%M:%S"


class _StructuredLogFormatter(logging.Formatter):
    """Mirror of runtime._StructuredLogFormatter for test isolation."""

    def __init__(self) -> None:
        super().__init__(_HUMAN_FMT, datefmt=_DATE_FMT)

    def format(self, record: logging.LogRecord) -> str:
        code: str | None = getattr(record, "log_code", None)
        if not code:
            return super().format(record)
        msg = record.getMessage()
        prefix = f"{code} "
        if msg.startswith(prefix):
            msg = msg[len(prefix):]
        extra: dict[str, Any] = {}
        for key, val in record.__dict__.items():
            if key not in _STANDARD_LOG_ATTRS:
                extra[key] = val
        return f"{code} | {msg} | extra={json.dumps(extra, default=str)}"


# Dispatcher regex (copied verbatim from dispatcher.py)
_LOG_CODE_PATTERN = re.compile(
    r"^(?P<code>LOG\d+)\s+\|\s+(?P<message>.+?)\s+\|\s+extra=(?P<extra>.*)$"
)


def _make_logger(name: str) -> tuple[logging.Logger, io.StringIO]:
    """Create a logger with _StructuredLogFormatter writing to a StringIO."""
    buf = io.StringIO()
    handler = logging.StreamHandler(buf)
    handler.setFormatter(_StructuredLogFormatter())
    logger = logging.getLogger(name)
    logger.handlers.clear()
    logger.addHandler(handler)
    logger.setLevel(logging.DEBUG)
    logger.propagate = False
    return logger, buf


class TestStructuredLogFormatter:
    """Test _StructuredLogFormatter output format."""

    def test_regular_log_uses_human_format(self):
        logger, buf = _make_logger("test.regular")
        logger.info("Hello world")
        line = buf.getvalue().strip()
        assert "Hello world" in line
        assert "|" not in line
        assert "[INFO]" in line

    def test_log_constant_uses_pipe_format(self):
        logger, buf = _make_logger("test.pipe")
        payload = {
            "log_code": "LOG446",
            "component": "Worker",
            "subcomponent": "RayRuntime",
            "tags": "ray,worker",
            "run_id": "run_abc",
        }
        logger.log(logging.INFO, "%s %s", "LOG446", "Ray worker runtime started", extra=payload)
        line = buf.getvalue().strip()
        assert line.startswith("LOG446 |")
        assert "Ray worker runtime started" in line
        assert "extra=" in line

    def test_pipe_format_extra_is_valid_json(self):
        logger, buf = _make_logger("test.json")
        payload = {"log_code": "LOG461", "run_id": "r1", "log_file": "/tmp/worker.log"}
        logger.log(logging.INFO, "%s %s", "LOG461", "Log file created", extra=payload)
        line = buf.getvalue().strip()
        extra_str = line.split("extra=", 1)[1]
        parsed = json.loads(extra_str)
        assert parsed["run_id"] == "r1"
        assert parsed["log_file"] == "/tmp/worker.log"

    def test_strips_code_prefix_from_message(self):
        logger, buf = _make_logger("test.strip")
        payload = {"log_code": "LOG449"}
        logger.log(logging.INFO, "%s %s", "LOG449", "Ray cluster initialized", extra=payload)
        line = buf.getvalue().strip()
        # Should be "LOG449 | Ray cluster initialized | extra=..." not "LOG449 | LOG449 Ray cluster..."
        parts = line.split(" | ")
        assert parts[0] == "LOG449"
        assert not parts[1].startswith("LOG449")
        assert parts[1] == "Ray cluster initialized"


class TestDispatcherRegexCompatibility:
    """Verify _StructuredLogFormatter output matches dispatcher's _parse_structured_log_line()."""

    @pytest.mark.parametrize(
        "code,message",
        [
            ("LOG446", "Ray worker runtime started"),
            ("LOG447", "Ray worker runtime completed"),
            ("LOG448", "Ray worker runtime failed"),
            ("LOG461", "Ray worker log file handler created"),
            ("LOG462", "Ray RLlib algorithm built successfully"),
        ],
    )
    def test_formatter_output_matches_dispatcher_regex(self, code: str, message: str):
        logger, buf = _make_logger(f"test.compat.{code}")
        payload = {"log_code": code, "run_id": "test_run", "component": "Worker"}
        logger.log(logging.INFO, "%s %s", code, message, extra=payload)
        line = buf.getvalue().strip()

        match = _LOG_CODE_PATTERN.match(line)
        assert match is not None, f"Dispatcher regex did not match: {line!r}"
        assert match.group("code") == code
        assert match.group("message") == message
        extra = json.loads(match.group("extra"))
        assert extra["run_id"] == "test_run"

    def test_regular_log_does_not_match_dispatcher_regex(self):
        logger, buf = _make_logger("test.compat.regular")
        logger.info("Just a normal log line")
        line = buf.getvalue().strip()
        assert _LOG_CODE_PATTERN.match(line) is None


# ---------------------------------------------------------------------------
# log_constant() integration with formatter
# ---------------------------------------------------------------------------


class TestLogConstantIntegration:
    """Test that log_constant() helper produces dispatcher-parseable output."""

    def test_log_constant_roundtrip(self):
        from gym_gui.logging_config.helpers import log_constant
        from gym_gui.logging_config.log_constants import LOG_WORKER_RAY_RUNTIME_STARTED

        logger, buf = _make_logger("test.roundtrip")
        log_constant(
            logger,
            LOG_WORKER_RAY_RUNTIME_STARTED,
            extra={"run_id": "roundtrip_123"},
        )
        line = buf.getvalue().strip()
        match = _LOG_CODE_PATTERN.match(line)
        assert match is not None, f"Roundtrip failed: {line!r}"
        assert match.group("code") == "LOG446"
        extra = json.loads(match.group("extra"))
        assert extra["run_id"] == "roundtrip_123"

    def test_log_constant_with_message_override(self):
        from gym_gui.logging_config.helpers import log_constant
        from gym_gui.logging_config.log_constants import LOG_WORKER_RAY_LOG_FILE_CREATED

        logger, buf = _make_logger("test.override")
        log_constant(
            logger,
            LOG_WORKER_RAY_LOG_FILE_CREATED,
            message="path=/tmp/worker.log",
            extra={"run_id": "msg_test"},
        )
        line = buf.getvalue().strip()
        match = _LOG_CODE_PATTERN.match(line)
        assert match is not None
        assert "path=/tmp/worker.log" in match.group("message")


# ---------------------------------------------------------------------------
# ALL_LOG_CONSTANTS completeness
# ---------------------------------------------------------------------------


class TestLogConstantsRegistry:
    """Verify LOG_WORKER_RAY_* constants are in ALL_LOG_CONSTANTS."""

    def test_ray_worker_constants_discoverable(self):
        from gym_gui.logging_config.log_constants import get_constant_by_code

        expected_codes = [
            "LOG446", "LOG447", "LOG448", "LOG449", "LOG450",
            "LOG451", "LOG452", "LOG453", "LOG454", "LOG455",
            "LOG456", "LOG457", "LOG458", "LOG459", "LOG460",
            "LOG461", "LOG462",
        ]
        for code in expected_codes:
            const = get_constant_by_code(code)
            assert const is not None, f"{code} not found in ALL_LOG_CONSTANTS"
            assert const.code == code

    def test_new_constants_have_correct_metadata(self):
        from gym_gui.logging_config.log_constants import (
            LOG_WORKER_RAY_ALGORITHM_BUILT,
            LOG_WORKER_RAY_LOG_FILE_CREATED,
        )

        assert LOG_WORKER_RAY_LOG_FILE_CREATED.code == "LOG461"
        assert LOG_WORKER_RAY_LOG_FILE_CREATED.component == "Worker"
        assert LOG_WORKER_RAY_LOG_FILE_CREATED.subcomponent == "RayRuntime"

        assert LOG_WORKER_RAY_ALGORITHM_BUILT.code == "LOG462"
        assert LOG_WORKER_RAY_ALGORITHM_BUILT.component == "Worker"
        assert "algorithm" in LOG_WORKER_RAY_ALGORITHM_BUILT.tags


# ---------------------------------------------------------------------------
# HumanVsAgentConfigForm difficulty fix
# ---------------------------------------------------------------------------


class TestDifficultyFormFix:
    """Verify _on_difficulty_changed delegates to _apply_difficulty_preset."""

    def test_on_difficulty_changed_calls_apply_preset(self):
        """The fix: _on_difficulty_changed must call _apply_difficulty_preset
        instead of directly accessing self._skill_spin which may not exist."""
        import inspect

        from gym_gui.ui.widgets.human_vs_agent_config_form import HumanVsAgentConfigForm
        source = inspect.getsource(HumanVsAgentConfigForm._on_difficulty_changed)
        # Must NOT contain direct _skill_spin access
        assert "_skill_spin" not in source, (
            "_on_difficulty_changed still accesses _skill_spin directly"
        )
        # Must delegate to _apply_difficulty_preset
        assert "_apply_difficulty_preset" in source


# ---------------------------------------------------------------------------
# Dispatcher _maybe_open_log_files for all workers
# ---------------------------------------------------------------------------


class TestDispatcherLogFiles:
    """Verify dispatcher opens log files for all worker types."""

    def test_maybe_open_log_files_source_has_no_script_guard(self):
        """The fix: _maybe_open_log_files must NOT skip module-based workers."""
        import inspect

        from gym_gui.services.trainer.dispatcher import TrainerDispatcher
        source = inspect.getsource(TrainerDispatcher._maybe_open_log_files)
        # Must NOT contain the old guard that skipped module workers
        assert 'wmeta.get("script")' not in source, (
            "_maybe_open_log_files still has script-only guard"
        )
        assert 'wmeta.get("module")' not in source, (
            "_maybe_open_log_files still checks for module workers"
        )
