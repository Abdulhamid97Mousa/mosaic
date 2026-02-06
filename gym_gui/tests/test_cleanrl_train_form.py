"""Tests for CleanRL training form."""

from __future__ import annotations

import os
from typing import Dict, Any

import pytest
from qtpy import QtWidgets

from gym_gui.core.enums import GameId
from gym_gui.telemetry.semconv import VideoModes
from gym_gui.ui.widgets.cleanrl_train_form import CleanRlTrainForm

# Ensure Qt renders offscreen in CI environments
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


@pytest.fixture(scope="module")
def qt_app():
    app = QtWidgets.QApplication.instance()
    if app is None:
        app = QtWidgets.QApplication([])
    yield app


def _base_form(qt_app) -> CleanRlTrainForm:
    form = CleanRlTrainForm(default_game=GameId.FROZEN_LAKE_V2)
    index = form._algo_combo.findText("ppo")
    if index >= 0:
        form._algo_combo.setCurrentIndex(index)
    env_index = form._env_combo.findData("CartPole-v1")
    if env_index < 0:
        env_index = 0
    form._env_combo.setCurrentIndex(env_index)
    form._test_selected_env = form._env_combo.currentData()
    form._timesteps_spin.setValue(4096)
    form._seed_spin.setValue(123)
    form._agent_id_input.setText("agent-cleanrl-test")
    form._worker_id_input.setText("cleanrl-test-worker")
    form._tensorboard_checkbox.setChecked(True)
    form._track_wandb_checkbox.setChecked(False)
    form._notes_edit.setPlainText("integration-test")
    return form


def test_get_config_includes_worker_metadata(qt_app) -> None:
    form = _base_form(qt_app)
    config = form.get_config()

    assert isinstance(config, dict)
    metadata = config.get("metadata", {})
    worker_meta: Dict[str, Any] = metadata.get("worker", {})
    assert worker_meta.get("module") == "cleanrl_worker.cli"
    assert worker_meta.get("worker_id") == "cleanrl-test-worker"
    assert worker_meta.get("config", {}).get("algo") == "ppo"
    assert worker_meta.get("config", {}).get("env_id") == form._test_selected_env
    assert worker_meta.get("config", {}).get("seed") == 123
    # arguments key is no longer present (removed dry-run checkbox)
    assert "arguments" not in worker_meta

    extras = worker_meta.get("config", {}).get("extras", {})
    assert extras.get("cuda") is True
    assert extras.get("tensorboard_dir") == "tensorboard"
    assert extras.get("notes") == "integration-test"
    assert "algo_params" in extras

    artifacts = metadata.get("artifacts", {})
    tensorboard_meta = artifacts.get("tensorboard", {})
    assert tensorboard_meta.get("enabled") is True
    assert tensorboard_meta.get("relative_path").endswith("tensorboard")
    wandb_meta = artifacts.get("wandb", {})
    assert wandb_meta.get("enabled") is False

    form.deleteLater()


def test_wandb_fields_populate_extras_and_environment(qt_app) -> None:
    form = _base_form(qt_app)
    form._track_wandb_checkbox.setChecked(True)
    form._wandb_project_input.setText("MOSAIC")
    form._wandb_entity_input.setText("abdulhamid97mousa")
    form._wandb_run_name_input.setText("demo-run")
    form._wandb_api_key_input.setText("test-key-123")

    config = form.get_config()
    metadata = config["metadata"]
    worker_meta = metadata["worker"]
    extras = worker_meta["config"].get("extras", {})
    assert extras.get("track_wandb") is True
    assert extras.get("wandb_project_name") == "MOSAIC"
    assert extras.get("wandb_entity") == "abdulhamid97mousa"
    assert extras.get("wandb_run_name") == "demo-run"
    env_overrides = config["environment"]
    assert env_overrides.get("WANDB_API_KEY") == "test-key-123"

    form.deleteLater()


def test_disable_gpu_sets_cuda_false(qt_app) -> None:
    form = _base_form(qt_app)
    form._use_gpu_checkbox.setChecked(False)

    config = form.get_config()
    extras = config["metadata"]["worker"]["config"].get("extras", {})
    assert extras.get("cuda") is False

    form.deleteLater()


def _num_env_spin(form: CleanRlTrainForm) -> QtWidgets.QSpinBox:
    widget = form._algo_param_inputs.get("num_envs")
    assert isinstance(widget, QtWidgets.QSpinBox)
    return widget


def test_video_mode_syncs_to_single_env(qt_app) -> None:
    form = _base_form(qt_app)
    num_envs = _num_env_spin(form)
    num_envs.setValue(1)

    assert form._video_mode_combo.currentData() == VideoModes.SINGLE
    assert form._grid_limit_spin.value() == 1
    assert form._fastlane_slot_spin.value() == 0

    form.deleteLater()


def test_video_mode_syncs_to_grid_when_multi_env(qt_app) -> None:
    form = _base_form(qt_app)
    num_envs = _num_env_spin(form)
    num_envs.setValue(4)

    assert form._video_mode_combo.currentData() == VideoModes.GRID
    assert form._grid_limit_spin.value() == 4
    assert 0 <= form._fastlane_slot_spin.value() <= 3

    form.deleteLater()


# =============================================================================
# Custom Script Tests
# =============================================================================


def test_custom_script_combo_exists_and_has_none_option(qt_app) -> None:
    """Test that the custom script dropdown exists with 'None' as default."""
    form = _base_form(qt_app)

    # Verify combo box exists
    assert hasattr(form, "_custom_script_combo")
    assert isinstance(form._custom_script_combo, QtWidgets.QComboBox)

    # First item should be "None (Standard Training)"
    assert form._custom_script_combo.count() >= 2  # At least None + Browse
    assert form._custom_script_combo.itemText(0) == "None (Standard Training)"
    assert form._custom_script_combo.itemData(0) is None

    # Default selection should be None
    assert form._custom_script_combo.currentIndex() == 0
    assert form._custom_script_combo.currentData() is None

    form.deleteLater()


def test_custom_script_combo_has_browse_option(qt_app) -> None:
    """Test that the custom script dropdown has 'Browse...' as last option."""
    form = _base_form(qt_app)

    last_index = form._custom_script_combo.count() - 1
    assert form._custom_script_combo.itemText(last_index) == "Browse..."
    assert form._custom_script_combo.itemData(last_index) == "BROWSE"

    form.deleteLater()


def test_custom_script_none_produces_standard_config(qt_app) -> None:
    """Test that selecting None produces standard cleanrl_worker config."""
    form = _base_form(qt_app)

    # Ensure None is selected
    form._custom_script_combo.setCurrentIndex(0)

    config = form.get_config()

    # Should use Python as entry point with cleanrl_worker.cli
    assert config["entry_point"].endswith("python") or "python" in config["entry_point"]
    assert config["arguments"] == ["-m", "cleanrl_worker.cli"]

    # Should not have MOSAIC_CONFIG_FILE in environment
    env = config.get("environment", {})
    assert "MOSAIC_CONFIG_FILE" not in env

    form.deleteLater()


def test_custom_script_populates_from_scripts_dir(qt_app) -> None:
    """Test that scripts from CLEANRL_SCRIPTS_DIR are loaded into dropdown."""
    from gym_gui.config.paths import CLEANRL_SCRIPTS_DIR

    form = _base_form(qt_app)

    # If scripts directory exists and has .sh files, they should appear
    if CLEANRL_SCRIPTS_DIR.is_dir():
        scripts = list(CLEANRL_SCRIPTS_DIR.glob("*.sh"))
        if scripts:
            # Should have more than just "None" and "Browse..."
            assert form._custom_script_combo.count() > 2

            # Find a script in the combo
            found_script = False
            for i in range(1, form._custom_script_combo.count() - 1):  # Skip None and Browse
                data = form._custom_script_combo.itemData(i)
                if data and data != "BROWSE":
                    found_script = True
                    assert data.endswith(".sh")
                    break
            assert found_script, "Expected at least one .sh script in dropdown"

    form.deleteLater()


def test_custom_script_selected_changes_config_entry_point(qt_app) -> None:
    """Test that selecting a script changes the config entry point to bash."""
    import tempfile
    from pathlib import Path

    form = _base_form(qt_app)

    # Create a temporary script file
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".sh", delete=False, prefix="test_script_"
    ) as f:
        f.write("#!/bin/bash\n# @description: Test script\necho 'test'\n")
        script_path = f.name

    try:
        # Manually add the script to the combo (simulating import)
        insert_index = form._custom_script_combo.count() - 1
        form._custom_script_combo.insertItem(insert_index, "test_script (imported)", script_path)
        form._custom_script_combo.setCurrentIndex(insert_index)

        config = form.get_config()

        # Should use bash as entry point
        assert config["entry_point"] == "/bin/bash"
        assert config["arguments"] == [script_path]

        # Should have MOSAIC environment variables
        env = config.get("environment", {})
        assert "MOSAIC_CONFIG_FILE" in env
        assert "MOSAIC_RUN_ID" in env
        assert "MOSAIC_CUSTOM_SCRIPTS_DIR" in env
        assert "MOSAIC_CHECKPOINT_DIR" in env

        # Metadata should reflect script mode
        metadata = config.get("metadata", {})
        ui_meta = metadata.get("ui", {})
        assert ui_meta.get("custom_script") == script_path
        assert "test_script" in ui_meta.get("custom_script_name", "")

    finally:
        Path(script_path).unlink(missing_ok=True)
        form.deleteLater()


def test_custom_script_config_file_written(qt_app) -> None:
    """Test that selecting a script causes config to be written to file."""
    import tempfile
    from pathlib import Path
    from gym_gui.config.paths import VAR_CUSTOM_SCRIPTS_DIR

    form = _base_form(qt_app)

    # Create a temporary script file
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".sh", delete=False, prefix="test_script_"
    ) as f:
        f.write("#!/bin/bash\n# @description: Test script\necho 'test'\n")
        script_path = f.name

    try:
        # Add and select the script
        insert_index = form._custom_script_combo.count() - 1
        form._custom_script_combo.insertItem(insert_index, "test_script (imported)", script_path)
        form._custom_script_combo.setCurrentIndex(insert_index)

        config = form.get_config()

        # Extract run_id and check config file exists
        run_id = config["run_name"]
        config_file_path = VAR_CUSTOM_SCRIPTS_DIR / run_id / "base_config.json"

        assert config_file_path.exists(), f"Config file should exist at {config_file_path}"

        # Verify config file is valid JSON with expected keys
        import json
        written_config = json.loads(config_file_path.read_text())
        assert "run_id" in written_config
        assert "algo" in written_config
        assert "env_id" in written_config

        # Cleanup
        import shutil
        shutil.rmtree(VAR_CUSTOM_SCRIPTS_DIR / run_id, ignore_errors=True)

    finally:
        Path(script_path).unlink(missing_ok=True)
        form.deleteLater()


def test_parse_script_metadata_extracts_description(qt_app) -> None:
    """Test that script metadata parsing extracts @description correctly."""
    import tempfile
    from pathlib import Path

    form = _base_form(qt_app)

    # Create a script with metadata
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".sh", delete=False, prefix="meta_script_"
    ) as f:
        f.write("#!/bin/bash\n")
        f.write("# my_curriculum.sh\n")
        f.write("#\n")
        f.write("# @description: DoorKey curriculum (5x5 → 8x8 → 16x16)\n")
        f.write("# @phases: 3\n")
        f.write("echo 'running'\n")
        script_path = Path(f.name)

    try:
        description = form._parse_script_metadata(script_path)
        assert "DoorKey" in description
        assert "curriculum" in description.lower()
    finally:
        script_path.unlink(missing_ok=True)
        form.deleteLater()


def test_script_validation_checks_file_exists(qt_app) -> None:
    """Test that script validation fails for non-existent script."""
    form = _base_form(qt_app)

    # Add a non-existent script path
    insert_index = form._custom_script_combo.count() - 1
    form._custom_script_combo.insertItem(
        insert_index, "fake_script", "/nonexistent/path/script.sh"
    )
    form._custom_script_combo.setCurrentIndex(insert_index)

    # Trigger validation via _on_validate_clicked (which calls _run_validation)
    # We'll directly test _run_script_validation
    state = form._collect_state()
    run_id = "test-validation-run"
    config = form._build_config(state, run_id=run_id)

    success, _ = form._run_script_validation(state, config, run_id, persist_config=False)

    assert success is False
    # Check that error is in the notes
    notes_text = form._notes_edit.toPlainText()
    assert "FAILED" in notes_text
    assert "not found" in notes_text.lower()

    form.deleteLater()


def test_script_validation_succeeds_for_valid_script(qt_app) -> None:
    """Test that script validation passes for a valid script."""
    import tempfile
    from pathlib import Path

    form = _base_form(qt_app)

    # Create a valid script
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".sh", delete=False, prefix="valid_script_"
    ) as f:
        f.write("#!/bin/bash\n")
        f.write("# @description: Test curriculum script\n")
        f.write('CONFIG="$MOSAIC_CONFIG_FILE"\n')
        f.write("python -m cleanrl_worker.cli --config $CONFIG\n")
        script_path = f.name

    try:
        # Add the script
        insert_index = form._custom_script_combo.count() - 1
        form._custom_script_combo.insertItem(insert_index, "valid_script", script_path)
        form._custom_script_combo.setCurrentIndex(insert_index)

        state = form._collect_state()
        run_id = "test-valid-script-run"
        config = form._build_config(state, run_id=run_id)

        success, _ = form._run_script_validation(state, config, run_id, persist_config=False)

        assert success is True
        notes_text = form._notes_edit.toPlainText()
        assert "SUCCESS" in notes_text
        assert "[PASSED] Custom Script Validation" in notes_text

    finally:
        Path(script_path).unlink(missing_ok=True)
        form.deleteLater()


def test_script_validation_warns_missing_mosaic_config_ref(qt_app) -> None:
    """Test that script validation warns if MOSAIC_CONFIG_FILE not referenced."""
    import tempfile
    from pathlib import Path

    form = _base_form(qt_app)

    # Create a script that doesn't reference MOSAIC_CONFIG_FILE
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".sh", delete=False, prefix="no_config_script_"
    ) as f:
        f.write("#!/bin/bash\n")
        f.write("# @description: Script without config reference\n")
        f.write("echo 'doing something'\n")
        script_path = f.name

    try:
        insert_index = form._custom_script_combo.count() - 1
        form._custom_script_combo.insertItem(insert_index, "no_config_script", script_path)
        form._custom_script_combo.setCurrentIndex(insert_index)

        state = form._collect_state()
        run_id = "test-warn-script-run"
        config = form._build_config(state, run_id=run_id)

        success, _ = form._run_script_validation(state, config, run_id, persist_config=False)

        # Should still succeed but with warning
        assert success is True
        notes_text = form._notes_edit.toPlainText()
        assert "SUCCESS" in notes_text
        assert "MOSAIC_CONFIG_FILE" in notes_text  # Warning about missing reference

    finally:
        Path(script_path).unlink(missing_ok=True)
        form.deleteLater()


def test_custom_script_sets_worker_metadata_for_dispatcher(qt_app) -> None:
    """Test that custom script mode sets metadata.worker correctly for dispatcher.

    The trainer dispatcher reads metadata.worker.module or metadata.worker.script
    to determine the subprocess command. When a custom script is selected:
    - metadata.worker.module should NOT be present (or dispatcher uses python -m)
    - metadata.worker.script should be set to '/bin/bash'
    - metadata.worker.arguments should contain the script path

    This is critical because the dispatcher ignores top-level entry_point/arguments.
    """
    import tempfile
    from pathlib import Path

    form = _base_form(qt_app)

    # Create a temporary script file
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".sh", delete=False, prefix="test_dispatcher_script_"
    ) as f:
        f.write("#!/bin/bash\n")
        f.write("# @description: Test script for dispatcher\n")
        f.write('CONFIG="$MOSAIC_CONFIG_FILE"\n')
        f.write("python -m cleanrl_worker.cli --config $CONFIG\n")
        script_path = f.name

    try:
        # Add and select the script
        insert_index = form._custom_script_combo.count() - 1
        form._custom_script_combo.insertItem(insert_index, "test_dispatcher_script", script_path)
        form._custom_script_combo.setCurrentIndex(insert_index)

        config = form.get_config()
        metadata = config.get("metadata", {})
        worker_meta = metadata.get("worker", {})

        # CRITICAL: module should NOT be present when using a custom script
        # If module is present, dispatcher will run 'python -m module' instead of bash script
        assert "module" not in worker_meta, (
            "metadata.worker.module should not be present in custom script mode. "
            "Dispatcher will run 'python -m module' instead of the bash script!"
        )

        # script should be set to /bin/bash
        assert worker_meta.get("script") == "/bin/bash", (
            "metadata.worker.script should be '/bin/bash' for custom scripts"
        )

        # arguments should contain the script path
        assert script_path in worker_meta.get("arguments", []), (
            f"metadata.worker.arguments should contain script path '{script_path}'"
        )

    finally:
        Path(script_path).unlink(missing_ok=True)
        form.deleteLater()


def test_standard_training_uses_module_not_script(qt_app) -> None:
    """Test that standard training mode uses metadata.worker.module."""
    form = _base_form(qt_app)

    # Ensure None (standard training) is selected
    form._custom_script_combo.setCurrentIndex(0)

    config = form.get_config()
    metadata = config.get("metadata", {})
    worker_meta = metadata.get("worker", {})

    # In standard mode, module should be set
    assert worker_meta.get("module") == "cleanrl_worker.cli"
    # script should NOT be present
    assert "script" not in worker_meta or worker_meta.get("script") is None

    form.deleteLater()


def test_custom_script_does_not_set_cleanrl_num_envs(qt_app) -> None:
    """Test that custom script mode does NOT set CLEANRL_NUM_ENVS.

    The script is the source of truth for training parameters. If the form
    sets CLEANRL_NUM_ENVS=1 (default), the script's intended default
    (e.g., 4 for grid view) would be overridden.

    The script uses: export CLEANRL_NUM_ENVS="${CLEANRL_NUM_ENVS:-4}"
    If we don't set it, the script's default (4) takes effect.
    """
    import tempfile
    from pathlib import Path

    form = _base_form(qt_app)

    # Create a temporary script file
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".sh", delete=False, prefix="test_num_envs_script_"
    ) as f:
        f.write("#!/bin/bash\n")
        f.write("# @description: Test script for num_envs\n")
        f.write('CONFIG="$MOSAIC_CONFIG_FILE"\n')
        f.write('export CLEANRL_NUM_ENVS="${CLEANRL_NUM_ENVS:-4}"\n')
        f.write("python -m cleanrl_worker.cli --config $CONFIG\n")
        script_path = f.name

    try:
        # Add and select the script
        insert_index = form._custom_script_combo.count() - 1
        form._custom_script_combo.insertItem(insert_index, "test_num_envs_script", script_path)
        form._custom_script_combo.setCurrentIndex(insert_index)

        config = form.get_config()
        env = config.get("environment", {})

        # CRITICAL: CLEANRL_NUM_ENVS should NOT be in environment for custom scripts
        # This allows the script to use its own default (e.g., 4 for grid view)
        assert "CLEANRL_NUM_ENVS" not in env, (
            "CLEANRL_NUM_ENVS should NOT be set in custom script mode. "
            "The script is the source of truth and should set its own num_envs."
        )

        # But it SHOULD be present in standard training mode
        form._custom_script_combo.setCurrentIndex(0)  # Select None (standard training)
        standard_config = form.get_config()
        standard_env = standard_config.get("environment", {})
        assert "CLEANRL_NUM_ENVS" in standard_env, (
            "CLEANRL_NUM_ENVS SHOULD be set in standard training mode"
        )

    finally:
        Path(script_path).unlink(missing_ok=True)
        form.deleteLater()
