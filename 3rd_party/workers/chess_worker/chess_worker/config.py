"""Configuration for Chess Worker."""

from dataclasses import dataclass, field
from typing import Optional, Dict, Any


@dataclass
class ChessWorkerConfig:
    """Configuration for Chess LLM Worker.

    Attributes:
        run_id: Unique run identifier.
        env_name: Environment family (always "pettingzoo" for chess).
        task: Environment task (always "chess_v6").
        client_name: LLM client ("vllm", "openai", "anthropic").
        model_id: Model identifier.
        base_url: API base URL for vLLM or compatible API.
        api_key: API key (optional for local vLLM).
        temperature: Sampling temperature.
        max_tokens: Maximum tokens in response.
        max_retries: Max retries for invalid moves.
        max_dialog_turns: Max conversation turns per move.
    """
    run_id: str = ""
    env_name: str = "pettingzoo"
    task: str = "chess_v6"

    # LLM settings
    client_name: str = "vllm"
    model_id: str = "Qwen/Qwen2.5-1.5B-Instruct"
    base_url: str = "http://127.0.0.1:8000/v1"
    api_key: Optional[str] = None

    # Generation settings
    temperature: float = 0.3
    max_tokens: int = 256

    # Chess-specific settings
    max_retries: int = 3  # Max invalid move attempts before giving up
    max_dialog_turns: int = 10  # Max conversation turns per move

    # Telemetry
    telemetry_dir: str = "var/telemetry"

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "run_id": self.run_id,
            "env_name": self.env_name,
            "task": self.task,
            "client_name": self.client_name,
            "model_id": self.model_id,
            "base_url": self.base_url,
            "api_key": self.api_key,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "max_retries": self.max_retries,
            "max_dialog_turns": self.max_dialog_turns,
            "telemetry_dir": self.telemetry_dir,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ChessWorkerConfig":
        """Create from dictionary."""
        return cls(
            run_id=data.get("run_id", ""),
            env_name=data.get("env_name", "pettingzoo"),
            task=data.get("task", "chess_v6"),
            client_name=data.get("client_name", "vllm"),
            model_id=data.get("model_id", "Qwen/Qwen2.5-1.5B-Instruct"),
            base_url=data.get("base_url", "http://127.0.0.1:8000/v1"),
            api_key=data.get("api_key"),
            temperature=data.get("temperature", 0.3),
            max_tokens=data.get("max_tokens", 256),
            max_retries=data.get("max_retries", 3),
            max_dialog_turns=data.get("max_dialog_turns", 10),
            telemetry_dir=data.get("telemetry_dir", "var/telemetry"),
        )
