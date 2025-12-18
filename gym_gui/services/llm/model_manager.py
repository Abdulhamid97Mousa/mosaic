"""Model manager for automatic LLM download and vLLM server management.

Handles:
- HuggingFace authentication for gated models (Llama, etc.)
- Automatic model download to local cache
- vLLM server lifecycle management
- Model availability checking

References:
- HuggingFace Hub Auth: https://huggingface.co/docs/huggingface_hub/en/package_reference/authentication
- HuggingFace Hub Quickstart: https://huggingface.co/docs/huggingface_hub/en/quick-start
"""

from __future__ import annotations

import logging
import os
import subprocess
import threading
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Callable, Optional

from gym_gui.config.paths import VAR_MODELS_DIR, VAR_MODELS_HF_CACHE

_LOGGER = logging.getLogger(__name__)


class ModelStatus(Enum):
    """Status of a model."""
    NOT_DOWNLOADED = "not_downloaded"
    DOWNLOADING = "downloading"
    DOWNLOADED = "downloaded"
    LOADING = "loading"
    READY = "ready"
    ERROR = "error"


@dataclass
class ModelInfo:
    """Information about a model."""
    model_id: str
    display_name: str
    status: ModelStatus = ModelStatus.NOT_DOWNLOADED
    download_progress: float = 0.0
    error_message: Optional[str] = None
    local_path: Optional[Path] = None
    requires_auth: bool = True  # Most Llama models require HF auth
    size_gb: float = 0.0  # Approximate size in GB


# Approximate model sizes for VRAM estimation (4-bit quantized)
MODEL_SIZES_GB = {
    "meta-llama/Llama-3.2-1B-Instruct": 2.0,
    "meta-llama/Llama-3.2-3B-Instruct": 4.0,
    "mistralai/Mistral-7B-Instruct-v0.3": 8.0,
    "meta-llama/Llama-3.1-8B-Instruct": 10.0,
    "meta-llama/Llama-3.1-70B-Instruct": 45.0,
}


class HuggingFaceAuth:
    """Handle HuggingFace authentication.

    Supports multiple authentication methods:
    1. Direct token set via UI (highest priority)
    2. Environment variable HF_TOKEN
    3. Cached token from huggingface_hub login
    4. huggingface-cli login

    References:
    - https://huggingface.co/docs/huggingface_hub/en/package_reference/authentication
    """

    _ui_token: Optional[str] = None  # Token set by user in UI (highest priority)
    _token_path = Path.home() / ".cache" / "huggingface" / "token"

    @classmethod
    def get_token(cls) -> Optional[str]:
        """Get the HuggingFace token.

        Priority order:
        1. UI-provided token (set via set_token)
        2. Environment variable HF_TOKEN
        3. Cached token file (~/.cache/huggingface/token)

        Returns:
            Token string or None if not authenticated
        """
        # 1. UI-provided token takes highest priority
        if cls._ui_token:
            return cls._ui_token

        # 2. Check environment variable
        token = os.getenv("HF_TOKEN") or os.getenv("HUGGING_FACE_HUB_TOKEN")
        if token:
            return token

        # 3. Check cached token file (from huggingface-cli login)
        if cls._token_path.exists():
            try:
                return cls._token_path.read_text().strip()
            except Exception as e:
                _LOGGER.warning(f"Failed to read HF token: {e}")

        # 4. Try huggingface_hub library
        try:
            from huggingface_hub import HfFolder
            token = HfFolder.get_token()
            if token:
                return token
        except ImportError:
            pass
        except Exception as e:
            _LOGGER.debug(f"HfFolder.get_token failed: {e}")

        return None

    @classmethod
    def set_token(cls, token: str, persist: bool = True) -> None:
        """Set the HuggingFace token.

        Args:
            token: The HF token (hf_xxx format)
            persist: Whether to persist to cache and env var
        """
        cls._ui_token = token

        if persist:
            # Set environment variable for current session
            os.environ["HF_TOKEN"] = token

            # Use huggingface_hub login for proper persistence
            try:
                from huggingface_hub import login
                login(token=token, add_to_git_credential=False)
                _LOGGER.info("HuggingFace token saved via huggingface_hub.login()")
            except ImportError:
                # Fallback to manual file write
                cls._token_path.parent.mkdir(parents=True, exist_ok=True)
                cls._token_path.write_text(token)
                _LOGGER.info("HuggingFace token saved to cache file")
            except Exception as e:
                _LOGGER.warning(f"Failed to persist token via huggingface_hub: {e}")
                # Fallback to manual file write
                try:
                    cls._token_path.parent.mkdir(parents=True, exist_ok=True)
                    cls._token_path.write_text(token)
                except Exception as e2:
                    _LOGGER.error(f"Failed to save token to file: {e2}")

    @classmethod
    def is_authenticated(cls) -> bool:
        """Check if we have a valid HF token."""
        token = cls.get_token()
        return token is not None and len(token) > 10

    @classmethod
    def validate_token(cls, token: str) -> tuple[bool, str]:
        """Validate a HuggingFace token.

        Args:
            token: Token to validate

        Returns:
            Tuple of (is_valid, message)
        """
        if not token:
            return False, "Token is empty"

        if not token.startswith("hf_"):
            return False, "Token should start with 'hf_'"

        if len(token) < 20:
            return False, "Token is too short"

        # Try to validate with HuggingFace API
        try:
            from huggingface_hub import HfApi
            api = HfApi(token=token)
            user_info = api.whoami()
            username = user_info.get("name", "Unknown")
            return True, f"Authenticated as: {username}"
        except ImportError:
            return True, "Token format looks valid (huggingface_hub not installed for full validation)"
        except Exception as e:
            error_str = str(e)
            if "401" in error_str or "Invalid" in error_str.lower():
                return False, "Invalid token - please check and try again"
            return False, f"Validation failed: {error_str}"

    @classmethod
    def clear_token(cls) -> None:
        """Clear the cached token."""
        cls._ui_token = None

        if cls._token_path.exists():
            try:
                cls._token_path.unlink()
            except Exception as e:
                _LOGGER.warning(f"Failed to delete token file: {e}")

        if "HF_TOKEN" in os.environ:
            del os.environ["HF_TOKEN"]

        # Also logout via huggingface_hub
        try:
            from huggingface_hub import logout
            logout()
        except ImportError:
            pass
        except Exception as e:
            _LOGGER.debug(f"huggingface_hub logout failed: {e}")

    @classmethod
    def get_token_source(cls) -> str:
        """Get the source of the current token.

        Returns:
            Description of where the token came from
        """
        if cls._ui_token:
            return "User input (Chat panel)"
        if os.getenv("HF_TOKEN"):
            return "Environment variable (HF_TOKEN)"
        if os.getenv("HUGGING_FACE_HUB_TOKEN"):
            return "Environment variable (HUGGING_FACE_HUB_TOKEN)"
        if cls._token_path.exists():
            return "Cache file (~/.cache/huggingface/token)"
        return "Not authenticated"


class ProxyConfig:
    """Proxy configuration for network requests."""

    _enabled: bool = False
    _http_proxy: Optional[str] = None
    _https_proxy: Optional[str] = None

    @classmethod
    def is_enabled(cls) -> bool:
        """Check if proxy is enabled."""
        return cls._enabled

    @classmethod
    def get_http_proxy(cls) -> Optional[str]:
        """Get HTTP proxy URL."""
        if cls._enabled:
            return cls._http_proxy or os.getenv("HTTP_PROXY") or os.getenv("http_proxy")
        return None

    @classmethod
    def get_https_proxy(cls) -> Optional[str]:
        """Get HTTPS proxy URL."""
        if cls._enabled:
            return cls._https_proxy or os.getenv("HTTPS_PROXY") or os.getenv("https_proxy")
        return None

    @classmethod
    def set_proxy(cls, enabled: bool, http_proxy: str = "", https_proxy: str = "") -> None:
        """Set proxy configuration.

        Args:
            enabled: Whether to use proxy
            http_proxy: HTTP proxy URL (e.g., http://127.0.0.1:7890)
            https_proxy: HTTPS proxy URL (e.g., https://127.0.0.1:7890)
        """
        cls._enabled = enabled
        cls._http_proxy = http_proxy if http_proxy else None
        cls._https_proxy = https_proxy if https_proxy else None

        if enabled:
            # Set environment variables for requests/urllib
            if cls._http_proxy:
                os.environ["HTTP_PROXY"] = cls._http_proxy
                os.environ["http_proxy"] = cls._http_proxy
            if cls._https_proxy:
                os.environ["HTTPS_PROXY"] = cls._https_proxy
                os.environ["https_proxy"] = cls._https_proxy
            _LOGGER.info(f"Proxy enabled: HTTP={cls._http_proxy}, HTTPS={cls._https_proxy}")
        else:
            # Clear proxy environment variables
            for var in ["HTTP_PROXY", "http_proxy", "HTTPS_PROXY", "https_proxy"]:
                if var in os.environ:
                    del os.environ[var]
            _LOGGER.info("Proxy disabled")

    @classmethod
    def apply_to_environment(cls) -> None:
        """Apply current proxy settings to environment variables."""
        if cls._enabled:
            if cls._http_proxy:
                os.environ["HTTP_PROXY"] = cls._http_proxy
                os.environ["http_proxy"] = cls._http_proxy
            if cls._https_proxy:
                os.environ["HTTPS_PROXY"] = cls._https_proxy
                os.environ["https_proxy"] = cls._https_proxy

    @classmethod
    def get_status(cls) -> str:
        """Get proxy status string."""
        if not cls._enabled:
            return "Proxy disabled"
        parts = []
        if cls._http_proxy:
            parts.append(f"HTTP: {cls._http_proxy}")
        if cls._https_proxy:
            parts.append(f"HTTPS: {cls._https_proxy}")
        return " | ".join(parts) if parts else "Proxy enabled (no URLs set)"


class ModelDownloader:
    """Download models from HuggingFace Hub."""

    def __init__(self, cache_dir: Optional[Path] = None):
        self.cache_dir = cache_dir or VAR_MODELS_HF_CACHE
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def is_model_downloaded(self, model_id: str) -> bool:
        """Check if a model is already downloaded."""
        # Check in our custom cache directory
        model_dir = self._get_model_path(model_id)
        if model_dir.exists() and any(model_dir.iterdir()):
            return True

        # Also check standard HF cache
        try:
            from huggingface_hub import try_to_load_from_cache, scan_cache_dir

            # Check if model exists in any cache
            cache_info = scan_cache_dir()
            for repo in cache_info.repos:
                if repo.repo_id == model_id:
                    return True
        except ImportError:
            pass
        except Exception as e:
            _LOGGER.debug(f"Cache scan failed: {e}")

        return False

    def _get_model_path(self, model_id: str) -> Path:
        """Get the local path for a model."""
        # Convert model_id to safe directory name
        safe_name = model_id.replace("/", "--")
        return self.cache_dir / safe_name

    def download_model(
        self,
        model_id: str,
        progress_callback: Optional[Callable[[float, str], None]] = None,
    ) -> Path:
        """Download a model from HuggingFace Hub.

        Args:
            model_id: HuggingFace model ID (e.g., "meta-llama/Llama-3.1-8B-Instruct")
            progress_callback: Optional callback for progress updates (progress: 0-100, message: str)

        Returns:
            Path to the downloaded model

        Raises:
            RuntimeError: If download fails
        """
        try:
            from huggingface_hub import snapshot_download, HfApi
        except ImportError:
            raise RuntimeError(
                "huggingface_hub is required for model download. "
                "Install with: pip install huggingface_hub"
            )

        # Check authentication for gated models
        token = HuggingFaceAuth.get_token()

        if progress_callback:
            progress_callback(0, f"Preparing to download {model_id}...")

        try:
            # Set cache directory environment variable
            os.environ["HF_HOME"] = str(VAR_MODELS_DIR)
            os.environ["HUGGINGFACE_HUB_CACHE"] = str(self.cache_dir)

            # Apply proxy settings if enabled
            ProxyConfig.apply_to_environment()

            if progress_callback:
                proxy_status = ProxyConfig.get_status()
                progress_callback(5, f"Connecting to HuggingFace Hub... ({proxy_status})")

            # Download the model
            # Note: local_dir_use_symlinks is deprecated and no longer needed
            local_dir = snapshot_download(
                repo_id=model_id,
                cache_dir=str(self.cache_dir),
                token=token,
                local_dir=str(self._get_model_path(model_id)),
            )

            if progress_callback:
                progress_callback(100, f"Download complete: {model_id}")

            _LOGGER.info(f"Model downloaded to: {local_dir}")
            return Path(local_dir)

        except Exception as e:
            error_msg = str(e)

            # Check for common errors
            if "401" in error_msg or "Unauthorized" in error_msg:
                raise RuntimeError(
                    f"Authentication required for {model_id}. "
                    "Please provide your HuggingFace token."
                )
            elif "403" in error_msg or "Forbidden" in error_msg:
                raise RuntimeError(
                    f"Access denied to {model_id}. "
                    "Please accept the model license at huggingface.co and ensure your token has access."
                )
            elif "404" in error_msg or "Not Found" in error_msg:
                raise RuntimeError(f"Model not found: {model_id}")
            else:
                raise RuntimeError(f"Download failed: {error_msg}")


class VLLMServerManager:
    """Manage vLLM server process."""

    def __init__(self):
        self._process: Optional[subprocess.Popen] = None
        self._model_id: Optional[str] = None
        self._port: int = 8000
        self._lock = threading.Lock()

    @property
    def is_running(self) -> bool:
        """Check if the server is running."""
        with self._lock:
            if self._process is None:
                return False
            return self._process.poll() is None

    @property
    def current_model(self) -> Optional[str]:
        """Get the currently loaded model."""
        return self._model_id if self.is_running else None

    def start(
        self,
        model_id: str,
        model_path: Optional[Path] = None,
        port: int = 8000,
        dtype: str = "auto",
        max_model_len: Optional[int] = None,
        gpu_memory_utilization: float = 0.9,
    ) -> None:
        """Start the vLLM server with a model.

        Args:
            model_id: HuggingFace model ID
            model_path: Optional local path to model (uses model_id if not provided)
            port: Server port (default: 8000)
            dtype: Data type for model weights
            max_model_len: Maximum sequence length
            gpu_memory_utilization: GPU memory fraction to use
        """
        with self._lock:
            # Stop existing server if running
            if self._process is not None and self._process.poll() is None:
                self.stop()

            # Build command
            model_arg = str(model_path) if model_path else model_id
            cmd = [
                "vllm", "serve", model_arg,
                "--port", str(port),
                "--dtype", dtype,
                "--gpu-memory-utilization", str(gpu_memory_utilization),
            ]

            if max_model_len:
                cmd.extend(["--max-model-len", str(max_model_len)])

            # Set environment for HF cache
            env = os.environ.copy()
            env["HF_HOME"] = str(VAR_MODELS_DIR)
            env["HUGGINGFACE_HUB_CACHE"] = str(VAR_MODELS_HF_CACHE)

            token = HuggingFaceAuth.get_token()
            if token:
                env["HF_TOKEN"] = token

            _LOGGER.info(f"Starting vLLM server: {' '.join(cmd)}")

            try:
                self._process = subprocess.Popen(
                    cmd,
                    env=env,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                )
                self._model_id = model_id
                self._port = port

                # Wait for server to start (check endpoint)
                self._wait_for_ready(timeout=120)

            except FileNotFoundError:
                raise RuntimeError(
                    "vLLM not found. Install with: pip install vllm\n\n"
                    "Installation is in progress in the background.\n"
                    "Please wait 5-10 minutes and try again.\n\n"
                    "Or use OpenRouter (cloud) provider instead:\n"
                    "1. Switch Provider to 'OpenRouter'\n"
                    "2. Enter your API key from https://openrouter.ai/keys"
                )
            except Exception as e:
                self._process = None
                self._model_id = None
                raise RuntimeError(f"Failed to start vLLM server: {e}")

    def _wait_for_ready(self, timeout: int = 120) -> None:
        """Wait for the server to be ready."""
        import requests

        start_time = time.time()
        url = f"http://localhost:{self._port}/v1/models"

        while time.time() - start_time < timeout:
            try:
                response = requests.get(url, timeout=2)
                if response.status_code == 200:
                    _LOGGER.info("vLLM server is ready")
                    return
            except requests.exceptions.RequestException:
                pass

            # Check if process died
            if self._process and self._process.poll() is not None:
                stdout = self._process.stdout.read() if self._process.stdout else ""
                raise RuntimeError(f"vLLM server crashed: {stdout[-1000:]}")

            time.sleep(2)

        raise RuntimeError(f"vLLM server did not start within {timeout} seconds")

    def stop(self) -> None:
        """Stop the vLLM server."""
        with self._lock:
            if self._process is not None:
                _LOGGER.info("Stopping vLLM server...")
                self._process.terminate()
                try:
                    self._process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    self._process.kill()
                    self._process.wait()
                self._process = None
                self._model_id = None

    def get_logs(self, max_lines: int = 100) -> str:
        """Get recent server logs."""
        if self._process is None or self._process.stdout is None:
            return ""

        # Note: This is a simplified implementation
        # In production, you'd want to stream logs to a buffer
        return "Server logs not available in real-time mode"


class ModelManager:
    """High-level manager for model download and serving."""

    _instance: Optional["ModelManager"] = None

    def __init__(self):
        self.downloader = ModelDownloader()
        self.server = VLLMServerManager()
        self._models: dict[str, ModelInfo] = {}

    @classmethod
    def get_instance(cls) -> "ModelManager":
        """Get the singleton instance."""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def get_model_info(self, model_id: str, display_name: str = "") -> ModelInfo:
        """Get or create model info."""
        if model_id not in self._models:
            self._models[model_id] = ModelInfo(
                model_id=model_id,
                display_name=display_name or model_id,
            )
            # Check if already downloaded
            if self.downloader.is_model_downloaded(model_id):
                self._models[model_id].status = ModelStatus.DOWNLOADED
                self._models[model_id].local_path = self.downloader._get_model_path(model_id)

        return self._models[model_id]

    def ensure_model_ready(
        self,
        model_id: str,
        progress_callback: Optional[Callable[[float, str], None]] = None,
        auto_start_server: bool = True,
    ) -> Path:
        """Ensure a model is downloaded and optionally start the server.

        Args:
            model_id: HuggingFace model ID
            progress_callback: Optional callback for progress updates
            auto_start_server: Whether to start vLLM server after download

        Returns:
            Path to the model
        """
        info = self.get_model_info(model_id)

        # Download if needed
        if info.status == ModelStatus.NOT_DOWNLOADED:
            info.status = ModelStatus.DOWNLOADING
            try:
                path = self.downloader.download_model(model_id, progress_callback)
                info.status = ModelStatus.DOWNLOADED
                info.local_path = path
            except Exception as e:
                info.status = ModelStatus.ERROR
                info.error_message = str(e)
                raise

        # Start server if requested
        if auto_start_server and info.status == ModelStatus.DOWNLOADED:
            if not self.server.is_running or self.server.current_model != model_id:
                info.status = ModelStatus.LOADING
                if progress_callback:
                    progress_callback(0, "Starting vLLM server...")
                try:
                    self.server.start(model_id, info.local_path)
                    info.status = ModelStatus.READY
                    if progress_callback:
                        progress_callback(100, "Server ready!")
                except Exception as e:
                    info.status = ModelStatus.ERROR
                    info.error_message = str(e)
                    raise

        return info.local_path or self.downloader._get_model_path(model_id)

    def shutdown(self) -> None:
        """Shutdown the server."""
        self.server.stop()


__all__ = [
    "ModelStatus",
    "ModelInfo",
    "HuggingFaceAuth",
    "ModelDownloader",
    "VLLMServerManager",
    "ModelManager",
]
