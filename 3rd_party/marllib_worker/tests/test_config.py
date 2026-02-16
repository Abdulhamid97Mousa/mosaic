"""Tests for MARLlib worker configuration."""

from __future__ import annotations

import json

import pytest

from marllib_worker.config import MARLlibWorkerConfig, load_worker_config


# ------------------------------------------------------------------
# Construction & validation
# ------------------------------------------------------------------


class TestConfigValidation:
    """Verify __post_init__ catches invalid configurations."""

    def test_valid_config(self):
        cfg = MARLlibWorkerConfig(
            run_id="run-001",
            algo="mappo",
            environment_name="mpe",
            map_name="simple_spread",
        )
        assert cfg.run_id == "run-001"
        assert cfg.algo == "mappo"

    def test_missing_run_id(self):
        with pytest.raises(ValueError, match="run_id"):
            MARLlibWorkerConfig(
                run_id="",
                algo="mappo",
                environment_name="mpe",
                map_name="simple_spread",
            )

    def test_missing_algo(self):
        with pytest.raises(ValueError, match="algo"):
            MARLlibWorkerConfig(
                run_id="x",
                algo="",
                environment_name="mpe",
                map_name="simple_spread",
            )

    def test_missing_environment_name(self):
        with pytest.raises(ValueError, match="environment_name"):
            MARLlibWorkerConfig(
                run_id="x",
                algo="mappo",
                environment_name="",
                map_name="simple_spread",
            )

    def test_missing_map_name(self):
        with pytest.raises(ValueError, match="map_name"):
            MARLlibWorkerConfig(
                run_id="x",
                algo="mappo",
                environment_name="mpe",
                map_name="",
            )

    def test_unknown_algo(self):
        with pytest.raises(ValueError, match="Unknown algorithm"):
            MARLlibWorkerConfig(
                run_id="x",
                algo="nonexistent",
                environment_name="mpe",
                map_name="y",
            )

    def test_bad_share_policy(self):
        with pytest.raises(ValueError, match="share_policy"):
            MARLlibWorkerConfig(
                run_id="x",
                algo="mappo",
                environment_name="mpe",
                map_name="y",
                share_policy="bad",
            )

    def test_bad_core_arch(self):
        with pytest.raises(ValueError, match="core_arch"):
            MARLlibWorkerConfig(
                run_id="x",
                algo="mappo",
                environment_name="mpe",
                map_name="y",
                core_arch="transformer",
            )


# ------------------------------------------------------------------
# Serialisation round-trip
# ------------------------------------------------------------------


class TestConfigSerialization:
    """Verify to_dict / from_dict preserve values."""

    def _make(self, **overrides):
        defaults = dict(
            run_id="test-001",
            algo="mappo",
            environment_name="mpe",
            map_name="simple_spread",
            seed=42,
        )
        defaults.update(overrides)
        return MARLlibWorkerConfig(**defaults)

    def test_roundtrip(self):
        original = self._make()
        d = original.to_dict()
        restored = MARLlibWorkerConfig.from_dict(d)
        assert restored.run_id == original.run_id
        assert restored.algo == original.algo
        assert restored.seed == original.seed
        assert restored.environment_name == original.environment_name

    def test_from_dict_unknown_keys_go_to_extras(self):
        data = dict(
            run_id="x",
            algo="ippo",
            environment_name="mpe",
            map_name="y",
            custom_flag=True,
        )
        cfg = MARLlibWorkerConfig.from_dict(data)
        assert cfg.extras.get("custom_flag") is True

    def test_with_overrides_returns_new_config(self):
        cfg = self._make(seed=1)
        new = cfg.with_overrides(seed=99)
        assert new.seed == 99
        assert cfg.seed == 1  # original unchanged

    def test_with_overrides_skips_none(self):
        cfg = self._make(seed=1)
        new = cfg.with_overrides(seed=None)
        assert new.seed == 1  # unchanged because None is skipped

    def test_to_dict_is_json_serialisable(self):
        cfg = self._make(algo_params={"lr": 0.001}, env_params={"max_cycles": 50})
        text = json.dumps(cfg.to_dict())
        assert "mappo" in text

    def test_all_algos_accepted(self):
        from marllib_worker.registries import ALL_ALGORITHMS

        for algo in ALL_ALGORITHMS:
            cfg = MARLlibWorkerConfig(
                run_id="x",
                algo=algo,
                environment_name="mpe",
                map_name="y",
            )
            assert cfg.algo == algo


# ------------------------------------------------------------------
# Config loading
# ------------------------------------------------------------------


class TestConfigLoading:
    """Verify load_worker_config handles file formats."""

    def test_direct_format(self, tmp_path):
        data = dict(
            run_id="run-002",
            algo="qmix",
            environment_name="smac",
            map_name="3m",
            seed=123,
        )
        path = tmp_path / "config.json"
        path.write_text(json.dumps(data))
        cfg = load_worker_config(path)
        assert cfg.run_id == "run-002"
        assert cfg.algo == "qmix"

    def test_nested_gui_format(self, tmp_path):
        nested = {
            "metadata": {
                "worker": {
                    "config": {
                        "run_id": "nested-001",
                        "algo": "ippo",
                        "environment_name": "mpe",
                        "map_name": "simple_tag",
                    }
                }
            }
        }
        path = tmp_path / "config.json"
        path.write_text(json.dumps(nested))
        cfg = load_worker_config(path)
        assert cfg.run_id == "nested-001"
        assert cfg.algo == "ippo"
