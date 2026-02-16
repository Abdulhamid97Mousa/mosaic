"""Tests for MARLlib worker metadata and capabilities."""

from __future__ import annotations

import pytest


def _skip_if_no_gym_gui():
    try:
        from gym_gui.core.worker import WorkerMetadata, WorkerCapabilities  # noqa: F401

        return False
    except ImportError:
        return True


_SKIP = _skip_if_no_gym_gui()


class TestWorkerMetadata:
    """Verify get_worker_metadata() returns correct data."""

    @pytest.mark.skipif(_SKIP, reason="gym_gui not available")
    def test_returns_tuple_of_two(self):
        from marllib_worker import get_worker_metadata

        result = get_worker_metadata()
        assert isinstance(result, tuple)
        assert len(result) == 2

    @pytest.mark.skipif(_SKIP, reason="gym_gui not available")
    def test_metadata_fields(self):
        from gym_gui.core.worker import WorkerMetadata
        from marllib_worker import get_worker_metadata

        metadata, _ = get_worker_metadata()
        assert isinstance(metadata, WorkerMetadata)
        assert metadata.name == "MARLlib Worker"
        assert metadata.upstream_library == "marllib"
        assert metadata.upstream_version == "1.0.3"
        assert metadata.license == "MIT"

    @pytest.mark.skipif(_SKIP, reason="gym_gui not available")
    def test_capabilities_fields(self):
        from gym_gui.core.worker import WorkerCapabilities
        from marllib_worker import get_worker_metadata

        _, caps = get_worker_metadata()
        assert isinstance(caps, WorkerCapabilities)
        assert caps.worker_type == "marllib"
        assert caps.max_agents == 100
        assert caps.supports_checkpointing is True

    @pytest.mark.skipif(_SKIP, reason="gym_gui not available")
    def test_paradigms(self):
        from marllib_worker import get_worker_metadata

        _, caps = get_worker_metadata()
        assert "independent_learning" in caps.supported_paradigms
        assert "centralized_critic" in caps.supported_paradigms
        assert "value_decomposition" in caps.supported_paradigms

    @pytest.mark.skipif(_SKIP, reason="gym_gui not available")
    def test_env_families_include_key_envs(self):
        from marllib_worker import get_worker_metadata

        _, caps = get_worker_metadata()
        for env in ("mpe", "smac", "mamujoco", "hanabi", "overcooked"):
            assert env in caps.env_families, f"{env} missing from env_families"

    @pytest.mark.skipif(_SKIP, reason="gym_gui not available")
    def test_multi_agent(self):
        from marllib_worker import get_worker_metadata

        _, caps = get_worker_metadata()
        assert caps.is_multi_agent() is True
