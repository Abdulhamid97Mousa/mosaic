"""Tests for curriculum learning scripts.

These tests verify that:
1. Curriculum scripts exist and have proper structure
2. jq transformations correctly override algo to PPO
3. Script metadata is parseable
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest


# Scripts directory relative to cleanrl_worker package
SCRIPTS_DIR = Path(__file__).resolve().parent.parent / "cleanrl_worker" / "scripts"


def _get_curriculum_scripts() -> list[Path]:
    """Get all curriculum scripts from the scripts directory."""
    if not SCRIPTS_DIR.is_dir():
        return []
    return list(SCRIPTS_DIR.glob("curriculum_*.sh"))


class TestCurriculumScriptsExist:
    """Test that curriculum scripts exist and have proper structure."""

    def test_scripts_directory_exists(self) -> None:
        """Test that the scripts directory exists."""
        assert SCRIPTS_DIR.is_dir(), f"Scripts directory not found: {SCRIPTS_DIR}"

    def test_at_least_one_curriculum_script_exists(self) -> None:
        """Test that at least one curriculum script exists."""
        scripts = _get_curriculum_scripts()
        assert len(scripts) > 0, "No curriculum scripts found"

    @pytest.mark.parametrize(
        "script_path",
        _get_curriculum_scripts(),
        ids=lambda p: p.name,
    )
    def test_script_has_shebang(self, script_path: Path) -> None:
        """Test that each script has a proper bash shebang."""
        content = script_path.read_text()
        assert content.startswith("#!/bin/bash"), (
            f"Script {script_path.name} should start with #!/bin/bash"
        )

    @pytest.mark.parametrize(
        "script_path",
        _get_curriculum_scripts(),
        ids=lambda p: p.name,
    )
    def test_script_has_description_metadata(self, script_path: Path) -> None:
        """Test that each script has @description metadata."""
        content = script_path.read_text()
        assert "@description:" in content, (
            f"Script {script_path.name} should have @description metadata"
        )

    @pytest.mark.parametrize(
        "script_path",
        _get_curriculum_scripts(),
        ids=lambda p: p.name,
    )
    def test_script_references_mosaic_config_file(self, script_path: Path) -> None:
        """Test that each script references MOSAIC_CONFIG_FILE."""
        content = script_path.read_text()
        assert "MOSAIC_CONFIG_FILE" in content, (
            f"Script {script_path.name} should reference MOSAIC_CONFIG_FILE"
        )

    @pytest.mark.parametrize(
        "script_path",
        _get_curriculum_scripts(),
        ids=lambda p: p.name,
    )
    def test_script_uses_set_e(self, script_path: Path) -> None:
        """Test that each script uses 'set -e' for error handling."""
        content = script_path.read_text()
        assert "set -e" in content, (
            f"Script {script_path.name} should use 'set -e' for error handling"
        )


class TestJqAlgoOverride:
    """Test that jq correctly overrides algo to PPO.

    This is critical because:
    - The form may have algo="c51" (disabled field)
    - The script must override to algo="ppo" for BabyAI/MiniGrid compatibility
    - If jq fails, training will crash with 'KeyError: final_observation'
    """

    @pytest.fixture
    def sample_base_config(self, tmp_path: Path) -> Path:
        """Create a sample base_config.json like the form would generate."""
        config = {
            "run_id": "test-curriculum-run",
            "algo": "c51",  # Form's disabled algo field
            "env_id": "BabyAI-GoToRedBallNoDists-v0",
            "total_timesteps": 500000,
            "seed": 1,
            "extras": {
                "cuda": True,
                "tensorboard_dir": "tensorboard",
                "algo_params": {},
            },
        }
        config_path = tmp_path / "base_config.json"
        config_path.write_text(json.dumps(config, indent=2))
        return config_path

    def test_jq_is_installed(self) -> None:
        """Test that jq is available on the system."""
        result = subprocess.run(
            ["which", "jq"],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, "jq is not installed"

    def test_jq_overrides_algo_to_ppo(self, sample_base_config: Path) -> None:
        """Test that jq correctly overrides algo from c51 to ppo."""
        # This is the jq command used in the curriculum scripts
        jq_cmd = [
            "jq",
            '--arg', 'env', 'BabyAI-GoToRedBallNoDists-v0',
            '--argjson', 'steps', '200000',
            '.algo = "ppo" | .env_id = $env | .total_timesteps = $steps | .extras.algo_params.env_id = $env',
            str(sample_base_config),
        ]

        result = subprocess.run(jq_cmd, capture_output=True, text=True)

        assert result.returncode == 0, f"jq failed: {result.stderr}"

        # Parse the output
        output_config = json.loads(result.stdout)

        # Verify algo was overridden to ppo
        assert output_config["algo"] == "ppo", (
            f"algo should be 'ppo', got '{output_config['algo']}'"
        )

        # Verify env_id was set
        assert output_config["env_id"] == "BabyAI-GoToRedBallNoDists-v0"

        # Verify total_timesteps was set
        assert output_config["total_timesteps"] == 200000

        # Verify extras.algo_params.env_id was set
        assert output_config["extras"]["algo_params"]["env_id"] == "BabyAI-GoToRedBallNoDists-v0"

    def test_jq_preserves_other_fields(self, sample_base_config: Path) -> None:
        """Test that jq preserves fields not being modified."""
        jq_cmd = [
            "jq",
            '.algo = "ppo"',
            str(sample_base_config),
        ]

        result = subprocess.run(jq_cmd, capture_output=True, text=True)
        assert result.returncode == 0

        output_config = json.loads(result.stdout)

        # Check preserved fields
        assert output_config["run_id"] == "test-curriculum-run"
        assert output_config["seed"] == 1
        assert output_config["extras"]["cuda"] is True
        assert output_config["extras"]["tensorboard_dir"] == "tensorboard"


class TestCurriculumScriptAlgoOverride:
    """Test that each curriculum script properly overrides algo in jq commands."""

    @pytest.mark.parametrize(
        "script_path",
        _get_curriculum_scripts(),
        ids=lambda p: p.name,
    )
    def test_script_overrides_algo_to_ppo(self, script_path: Path) -> None:
        """Test that each script's jq commands include '.algo = \"ppo\"'.

        Note: jq commands often span multiple lines with backslash continuation,
        so we check the full script content, not line by line.
        """
        content = script_path.read_text()

        # Check that the script uses jq
        assert "jq " in content, (
            f"Script {script_path.name} should use jq to transform config"
        )

        # Check that the script overrides algo to ppo
        # The pattern appears in jq filter strings like: .algo = "ppo"
        assert '.algo = "ppo"' in content, (
            f"Script {script_path.name} should override algo to 'ppo' using jq.\n"
            f"Expected: .algo = \"ppo\" in a jq filter\n"
            f"This is required because BabyAI/MiniGrid environments only work with PPO, "
            f"not with value-based algorithms like c51/dqn."
        )

        # Count how many times the override appears (should match number of phases)
        override_count = content.count('.algo = "ppo"')
        assert override_count >= 1, (
            f"Script {script_path.name} should have at least one algo override"
        )

    @pytest.mark.parametrize(
        "script_path",
        _get_curriculum_scripts(),
        ids=lambda p: p.name,
    )
    def test_script_has_valid_bash_syntax(self, script_path: Path) -> None:
        """Test that each script has valid bash syntax."""
        result = subprocess.run(
            ["bash", "-n", str(script_path)],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, (
            f"Script {script_path.name} has bash syntax errors:\n{result.stderr}"
        )
