"""UI helpers for ALE (Atari) environment configuration panels."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict

from PyQt6 import QtWidgets

from gym_gui.config.game_configs import ALEConfig
from gym_gui.core.enums import GameId


ALE_GAME_IDS: tuple[GameId, ...] = (
    GameId.ADVENTURE_V4,
    GameId.ALE_ADVENTURE_V5,
    GameId.AIR_RAID_V4,
    GameId.ALE_AIR_RAID_V5,
    GameId.ASSAULT_V4,
    GameId.ALE_ASSAULT_V5,
    GameId.ALE_ALIEN_V5,
    GameId.ALE_AMIDAR_V5,
    GameId.ALE_ASTERIX_V5,
    GameId.ALE_ASTEROIDS_V5,
    GameId.ALE_ATLANTIS_V5,
    GameId.ALE_BACKGAMMON_V5,
    GameId.ALE_BANK_HEIST_V5,
    GameId.ALE_BASIC_MATH_V5,
    GameId.ALE_BATTLE_ZONE_V5,
    GameId.ALE_BEAM_RIDER_V5,
    GameId.ALE_BERZERK_V5,
    GameId.ALE_BLACKJACK_V5,
    GameId.ALE_BOWLING_V5,
    GameId.ALE_BOXING_V5,
    GameId.ALE_BREAKOUT_V5,
    GameId.ALE_CARNIVAL_V5,
    GameId.ALE_CASINO_V5,
    GameId.ALE_CENTIPEDE_V5,
    GameId.ALE_CHOPPER_COMMAND_V5,
    GameId.ALE_CRAZY_CLIMBER_V5,
    GameId.ALE_CROSSBOW_V5,
    GameId.ALE_DARKCHAMBERS_V5,
    GameId.ALE_DEFENDER_V5,
    GameId.ALE_DEMON_ATTACK_V5,
    GameId.ALE_DONKEY_KONG_V5,
    GameId.ALE_DOUBLE_DUNK_V5,
    GameId.ALE_EARTHWORLD_V5,
    GameId.ALE_ELEVATOR_ACTION_V5,
    GameId.ALE_ENDURO_V5,
    GameId.ALE_ENTOMBED_V5,
    GameId.ALE_ET_V5,
    GameId.ALE_FISHING_DERBY_V5,
    GameId.ALE_FLAG_CAPTURE_V5,
    GameId.ALE_FREEWAY_V5,
    GameId.ALE_FROGGER_V5,
    GameId.ALE_FROSTBITE_V5,
    GameId.ALE_GALAXIAN_V5,
    GameId.ALE_GOPHER_V5,
    GameId.ALE_GRAVITAR_V5,
    GameId.ALE_HANGMAN_V5,
    GameId.ALE_HAUNTED_HOUSE_V5,
    GameId.ALE_HERO_V5,
    GameId.ALE_HUMAN_CANNONBALL_V5,
    GameId.ALE_ICE_HOCKEY_V5,
    GameId.ALE_JAMESBOND_V5,
    GameId.ALE_JOURNEY_ESCAPE_V5,
    GameId.ALE_KABOOM_V5,
    GameId.ALE_KANGAROO_V5,
    GameId.ALE_KEYSTONE_KAPERS_V5,
    GameId.ALE_KING_KONG_V5,
    GameId.ALE_KLAX_V5,
    GameId.ALE_KOOLAID_V5,
    GameId.ALE_KRULL_V5,
    GameId.ALE_KUNG_FU_MASTER_V5,
    GameId.ALE_LASER_GATES_V5,
    GameId.ALE_LOST_LUGGAGE_V5,
    GameId.ALE_MARIO_BROS_V5,
    GameId.ALE_MINIATURE_GOLF_V5,
    GameId.ALE_MONTEZUMA_REVENGE_V5,
    GameId.ALE_MR_DO_V5,
    GameId.ALE_MS_PACMAN_V5,
    GameId.ALE_NAME_THIS_GAME_V5,
    GameId.ALE_OTHELLO_V5,
    GameId.ALE_PACMAN_V5,
    GameId.ALE_PHOENIX_V5,
    GameId.ALE_PITFALL_V5,
    GameId.ALE_PONG_V5,
    GameId.ALE_POOYAN_V5,
    GameId.ALE_PRIVATE_EYE_V5,
    GameId.ALE_QBERT_V5,
    GameId.ALE_RIVERRAID_V5,
    GameId.ALE_ROAD_RUNNER_V5,
    GameId.ALE_ROBOTANK_V5,
    GameId.ALE_SEAQUEST_V5,
    GameId.ALE_SIR_LANCELOT_V5,
    GameId.ALE_SKIING_V5,
    GameId.ALE_SOLARIS_V5,
    GameId.ALE_SPACE_INVADERS_V5,
    GameId.ALE_SPACE_WAR_V5,
    GameId.ALE_STAR_GUNNER_V5,
    GameId.ALE_SUPERMAN_V5,
    GameId.ALE_SURROUND_V5,
    GameId.ALE_TENNIS_V5,
    GameId.ALE_TETRIS_V5,
    GameId.ALE_TIME_PILOT_V5,
    GameId.ALE_TRONDEAD_V5,
    GameId.ALE_TURMOIL_V5,
    GameId.ALE_TUTANKHAM_V5,
    GameId.ALE_UP_N_DOWN_V5,
    GameId.ALE_VENTURE_V5,
    GameId.ALE_VIDEO_CHECKERS_V5,
    GameId.ALE_VIDEO_CHESS_V5,
    GameId.ALE_VIDEO_CUBE_V5,
    GameId.ALE_VIDEO_PINBALL_V5,
    GameId.ALE_WIZARD_OF_WOR_V5,
    GameId.ALE_WORD_ZAPPER_V5,
    GameId.ALE_YARS_REVENGE_V5,
    GameId.ALE_ZAXXON_V5,
)


@dataclass(slots=True)
class ControlCallbacks:
    """Callback container used to notify control panel of config changes."""

    on_change: Callable[[str, Any], None]


def build_ale_controls(
    *,
    parent: QtWidgets.QWidget,
    layout: QtWidgets.QFormLayout,
    game_id: GameId,
    overrides: Dict[str, Any],
    defaults: ALEConfig | None = None,
    callbacks: ControlCallbacks | None = None,
) -> None:
    """Populate ALE-specific controls into the provided layout.

    Exposes common ALE configuration:
    - obs_type: rgb | ram | grayscale
    - frameskip: single int or range (min,max)
    - repeat_action_probability (RAP)
    - difficulty and mode (flavour)
    - full_action_space
    """

    def emit_change(key: str, value: Any) -> None:
        if callbacks is not None:
            callbacks.on_change(key, value)

    # -------- obs_type --------
    obs_default = (defaults.obs_type if isinstance(defaults, ALEConfig) else overrides.get("obs_type", "rgb"))
    if not isinstance(obs_default, str):
        obs_default = "rgb"
    overrides["obs_type"] = obs_default
    obs_combo = QtWidgets.QComboBox(parent)
    obs_combo.addItems(["rgb", "ram", "grayscale"]) 
    index = obs_combo.findText(obs_default)
    obs_combo.setCurrentIndex(index if index >= 0 else 0)
    obs_combo.currentTextChanged.connect(lambda text: emit_change("obs_type", str(text)))
    obs_combo.setToolTip("Observation type from ALE: rgb, ram, or grayscale.")
    layout.addRow("Observation", obs_combo)

    # -------- full_action_space --------
    fas_value = bool(overrides.get("full_action_space", getattr(defaults, "full_action_space", False)))
    overrides["full_action_space"] = fas_value
    fas_checkbox = QtWidgets.QCheckBox("Request full 18-action space", parent)
    fas_checkbox.setChecked(fas_value)
    fas_checkbox.toggled.connect(lambda checked: emit_change("full_action_space", bool(checked)))
    layout.addRow("Action Space", fas_checkbox)

    # -------- frameskip --------
    fs_raw: Any = overrides.get("frameskip", getattr(defaults, "frameskip", None))
    # Model frameskip as either disabled (None), single int, or range via two spins
    use_range = isinstance(fs_raw, (tuple, list)) and len(fs_raw) == 2
    if use_range:
        fs_tuple = tuple(int(v) for v in list(fs_raw))  # type: ignore[arg-type]
        fs_min = fs_tuple[0]
        fs_max = fs_tuple[1]
    else:
        fs_min = int(fs_raw) if isinstance(fs_raw, (int, float)) else 4
        fs_max = fs_min

    range_checkbox = QtWidgets.QCheckBox("Use range (min,max)", parent)
    range_checkbox.setChecked(bool(use_range))

    fs_min_spin = QtWidgets.QSpinBox(parent)
    fs_min_spin.setRange(1, 10)
    fs_min_spin.setValue(int(fs_min))
    fs_max_spin = QtWidgets.QSpinBox(parent)
    fs_max_spin.setRange(1, 10)
    fs_max_spin.setValue(int(fs_max))

    def _emit_frameskip() -> None:
        if range_checkbox.isChecked():
            value: Any = (int(fs_min_spin.value()), int(fs_max_spin.value()))
        else:
            value = int(fs_min_spin.value())
        emit_change("frameskip", value)

    range_checkbox.toggled.connect(lambda _checked: _emit_frameskip())
    fs_min_spin.valueChanged.connect(lambda _val: _emit_frameskip())
    fs_max_spin.valueChanged.connect(lambda _val: _emit_frameskip())

    fs_row = QtWidgets.QWidget(parent)
    fs_row_layout = QtWidgets.QHBoxLayout(fs_row)
    fs_row_layout.setContentsMargins(0, 0, 0, 0)
    fs_row_layout.addWidget(range_checkbox)
    fs_row_layout.addWidget(QtWidgets.QLabel("min", parent))
    fs_row_layout.addWidget(fs_min_spin)
    fs_row_layout.addWidget(QtWidgets.QLabel("max", parent))
    fs_row_layout.addWidget(fs_max_spin)
    layout.addRow("Frameskip", fs_row)

    # -------- RAP (repeat_action_probability) --------
    rap_raw: Any = overrides.get("repeat_action_probability", getattr(defaults, "repeat_action_probability", None))
    try:
        rap_value = float(rap_raw) if rap_raw is not None else 0.25
    except (TypeError, ValueError):
        rap_value = 0.25
    overrides["repeat_action_probability"] = rap_value
    rap_spin = QtWidgets.QDoubleSpinBox(parent)
    rap_spin.setRange(0.0, 1.0)
    rap_spin.setSingleStep(0.05)
    rap_spin.setDecimals(2)
    rap_spin.setValue(rap_value)
    rap_spin.valueChanged.connect(lambda value: emit_change("repeat_action_probability", float(value)))
    layout.addRow("RAP", rap_spin)

    # -------- difficulty/mode --------
    diff_raw: Any = overrides.get("difficulty", getattr(defaults, "difficulty", None))
    mode_raw: Any = overrides.get("mode", getattr(defaults, "mode", None))
    diff_value = int(diff_raw) if isinstance(diff_raw, (int, float)) and int(diff_raw) >= 0 else 0
    mode_value = int(mode_raw) if isinstance(mode_raw, (int, float)) and int(mode_raw) >= 0 else 0
    overrides["difficulty"] = None if diff_value == 0 else diff_value
    overrides["mode"] = None if mode_value == 0 else mode_value

    diff_spin = QtWidgets.QSpinBox(parent)
    diff_spin.setRange(0, 9)
    diff_spin.setSpecialValueText("Default")
    diff_spin.setValue(diff_value)
    diff_spin.valueChanged.connect(
        lambda value: emit_change("difficulty", None if int(value) == 0 else int(value))
    )
    mode_spin = QtWidgets.QSpinBox(parent)
    mode_spin.setRange(0, 9)
    mode_spin.setSpecialValueText("Default")
    mode_spin.setValue(mode_value)
    mode_spin.valueChanged.connect(
        lambda value: emit_change("mode", None if int(value) == 0 else int(value))
    )

    layout.addRow("Difficulty", diff_spin)
    layout.addRow("Mode", mode_spin)

    # Guidance label
    guidance = QtWidgets.QLabel(
        "ALE v5 defaults: frameskip=4, RAP=0.25. Set a range for stochastic frame skip if desired.",
        parent,
    )
    guidance.setWordWrap(True)
    layout.addRow("", guidance)
