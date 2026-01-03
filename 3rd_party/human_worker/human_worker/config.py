"""Configuration for Human Worker."""

from dataclasses import dataclass
from typing import Any, Dict, Optional


@dataclass
class HumanWorkerConfig:
    """Configuration for Human Worker.

    Attributes:
        run_id: Unique run identifier.
        player_name: Human player's display name.
        timeout_seconds: Timeout for human input (0 = no timeout).
        show_legal_moves: Whether to highlight legal moves in UI.
        confirm_moves: Whether to require move confirmation.
    """

    run_id: str = ""
    player_name: str = "Human"

    # Timeout settings
    timeout_seconds: float = 0.0  # 0 = no timeout (wait forever)

    # UI hints
    show_legal_moves: bool = True
    confirm_moves: bool = False

    # Telemetry
    telemetry_dir: str = "var/telemetry"

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "run_id": self.run_id,
            "player_name": self.player_name,
            "timeout_seconds": self.timeout_seconds,
            "show_legal_moves": self.show_legal_moves,
            "confirm_moves": self.confirm_moves,
            "telemetry_dir": self.telemetry_dir,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "HumanWorkerConfig":
        """Create from dictionary."""
        return cls(
            run_id=data.get("run_id", ""),
            player_name=data.get("player_name", "Human"),
            timeout_seconds=data.get("timeout_seconds", 0.0),
            show_legal_moves=data.get("show_legal_moves", True),
            confirm_moves=data.get("confirm_moves", False),
            telemetry_dir=data.get("telemetry_dir", "var/telemetry"),
        )
