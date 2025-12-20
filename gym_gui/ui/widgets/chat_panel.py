"""Chat panel widget for LLM interaction.

Provides a chat interface with:
- API key input (with show/hide toggle)
- Model selection dropdown
- Chat history display
- Message input and send button

Uses QThread for async HTTP requests to avoid blocking the Qt event loop.
"""

from __future__ import annotations

import logging
from typing import List, Optional, TYPE_CHECKING

from qtpy import QtCore, QtGui, QtWidgets
from PyQt6.QtCore import pyqtSignal  # type: ignore[attr-defined]
from gym_gui.services.llm import (
    LLMConfig,
    LLMProvider,
    LLMService,
    ModelIdentity,
    ChatMessage,
    CompletionResult,
    HuggingFaceAuth,
    ProxyConfig,
    ModelManager,
    ModelStatus,
    GPUDetector,
    GPUDetectionResult,
    MODEL_SIZES_GB,
)
from gym_gui.logging_config.helpers import log_constant
from gym_gui.logging_config.log_constants import (
    LOG_UI_CHAT_GPU_DETECTION_STARTED,
    LOG_UI_CHAT_GPU_DETECTION_COMPLETED,
    LOG_UI_CHAT_GPU_DETECTION_ERROR,
    LOG_UI_CHAT_HF_TOKEN_SAVE_STARTED,
    LOG_UI_CHAT_HF_TOKEN_SAVED,
    LOG_UI_CHAT_HF_TOKEN_SAVE_ERROR,
    LOG_UI_CHAT_HF_TOKEN_VALIDATION_STARTED,
    LOG_UI_CHAT_HF_TOKEN_VALIDATED,
    LOG_UI_CHAT_HF_TOKEN_VALIDATION_ERROR,
    LOG_UI_CHAT_MODEL_DOWNLOAD_STARTED,
    LOG_UI_CHAT_MODEL_DOWNLOAD_PROGRESS,
    LOG_UI_CHAT_MODEL_DOWNLOADED,
    LOG_UI_CHAT_MODEL_DOWNLOAD_ERROR,
    LOG_UI_CHAT_REQUEST_STARTED,
    LOG_UI_CHAT_REQUEST_COMPLETED,
    LOG_UI_CHAT_REQUEST_ERROR,
    LOG_UI_CHAT_REQUEST_CANCELLED,
    LOG_UI_CHAT_PROXY_ENABLED,
    LOG_UI_CHAT_PROXY_DISABLED,
    LOG_UI_CHAT_CLEANUP_WARNING,
)

if TYPE_CHECKING:
    pass

_LOGGER = logging.getLogger(__name__)


class GPUDetectionWorker(QtCore.QThread):
    """Worker thread for GPU detection.

    Detects NVIDIA GPUs in the background to avoid blocking UI.
    """

    finished = pyqtSignal(object)  # GPUDetectionResult

    def __init__(self, parent: Optional[QtCore.QObject] = None) -> None:
        super().__init__(parent)

    def run(self) -> None:
        """Execute GPU detection in background thread."""
        log_constant(_LOGGER, LOG_UI_CHAT_GPU_DETECTION_STARTED)

        try:
            result = GPUDetector.detect()
            log_constant(
                _LOGGER,
                LOG_UI_CHAT_GPU_DETECTION_COMPLETED,
                extra={"gpu_count": len(result.gpus), "has_cuda": result.cuda_available},
            )
            self.finished.emit(result)
        except Exception as e:
            log_constant(
                _LOGGER,
                LOG_UI_CHAT_GPU_DETECTION_ERROR,
                message="GPU detection worker failed",
                exc_info=e,
            )
            # Emit empty result on error
            self.finished.emit(GPUDetectionResult(
                gpus=[],
                nvidia_smi_available=False,
                cuda_available=False,
                error_message=str(e),
            ))


class HFTokenValidationWorker(QtCore.QThread):
    """Worker thread for HuggingFace token validation.

    Validates token with HuggingFace API in background.
    """

    finished = pyqtSignal(bool, str)  # (is_valid, message)

    def __init__(
        self,
        token: str,
        parent: Optional[QtCore.QObject] = None,
    ) -> None:
        super().__init__(parent)
        self._token = token

    def run(self) -> None:
        """Validate token in background thread."""
        log_constant(_LOGGER, LOG_UI_CHAT_HF_TOKEN_VALIDATION_STARTED)

        try:
            is_valid, message = HuggingFaceAuth.validate_token(self._token)
            if is_valid:
                log_constant(_LOGGER, LOG_UI_CHAT_HF_TOKEN_VALIDATED, message=message)
            else:
                log_constant(_LOGGER, LOG_UI_CHAT_HF_TOKEN_VALIDATION_ERROR, message=message)
            self.finished.emit(is_valid, message)
        except Exception as e:
            log_constant(
                _LOGGER,
                LOG_UI_CHAT_HF_TOKEN_VALIDATION_ERROR,
                message="Token validation worker failed",
                exc_info=e,
            )
            self.finished.emit(False, f"Validation error: {e}")


class HFTokenSaveWorker(QtCore.QThread):
    """Worker thread for HuggingFace token saving.

    Saves token with HuggingFace Hub in background to avoid GUI freeze.
    The huggingface_hub.login() call makes network requests and can take
    several seconds depending on network conditions.
    """

    finished = pyqtSignal(bool, str)  # (success, message)

    def __init__(
        self,
        token: str,
        persist: bool = True,
        parent: Optional[QtCore.QObject] = None,
    ) -> None:
        super().__init__(parent)
        self._token = token
        self._persist = persist

    def run(self) -> None:
        """Save token in background thread."""
        log_constant(_LOGGER, LOG_UI_CHAT_HF_TOKEN_SAVE_STARTED)

        try:
            HuggingFaceAuth.set_token(self._token, persist=self._persist)
            log_constant(_LOGGER, LOG_UI_CHAT_HF_TOKEN_SAVED)
            self.finished.emit(True, "Token saved successfully!")
        except Exception as e:
            log_constant(
                _LOGGER,
                LOG_UI_CHAT_HF_TOKEN_SAVE_ERROR,
                message="Token save worker failed",
                exc_info=e,
            )
            self.finished.emit(False, f"Save error: {e}")


class ModelDownloadWorker(QtCore.QThread):
    """Worker thread for downloading LLM models from HuggingFace.

    Handles model download with progress updates.
    """

    progress = pyqtSignal(float, str)  # (percentage, message)
    finished = pyqtSignal(str)  # model path
    error = pyqtSignal(str)

    def __init__(
        self,
        model_id: str,
        parent: Optional[QtCore.QObject] = None,
    ) -> None:
        super().__init__(parent)
        self._model_id = model_id
        self._cancelled = False

    def run(self) -> None:
        """Execute the model download in background thread."""
        log_constant(
            _LOGGER,
            LOG_UI_CHAT_MODEL_DOWNLOAD_STARTED,
            extra={"model_id": self._model_id},
        )

        try:
            manager = ModelManager.get_instance()
            path = manager.ensure_model_ready(
                self._model_id,
                progress_callback=self._on_progress,
                auto_start_server=True,
            )
            if not self._cancelled:
                log_constant(
                    _LOGGER,
                    LOG_UI_CHAT_MODEL_DOWNLOADED,
                    extra={"model_id": self._model_id, "path": str(path)},
                )
                self.finished.emit(str(path))
        except Exception as e:
            if not self._cancelled:
                log_constant(
                    _LOGGER,
                    LOG_UI_CHAT_MODEL_DOWNLOAD_ERROR,
                    message=f"Model download failed: {self._model_id}",
                    exc_info=e,
                )
                self.error.emit(str(e))

    def _on_progress(self, percentage: float, message: str) -> None:
        """Forward progress to Qt signal."""
        if not self._cancelled:
            log_constant(
                _LOGGER,
                LOG_UI_CHAT_MODEL_DOWNLOAD_PROGRESS,
                extra={"model_id": self._model_id, "percentage": percentage, "status": message},
            )
            self.progress.emit(percentage, message)

    def cancel(self) -> None:
        """Mark as cancelled."""
        self._cancelled = True


class ChatWorker(QtCore.QThread):
    """Worker thread for LLM chat completion requests.

    Runs HTTP requests in a background thread to avoid blocking Qt UI.
    """

    finished = pyqtSignal(object)  # CompletionResult
    error = pyqtSignal(str)

    def __init__(
        self,
        service: LLMService,
        messages: List[ChatMessage],
        model: ModelIdentity,
        parent: Optional[QtCore.QObject] = None,
    ) -> None:
        super().__init__(parent)
        self._service = service
        self._messages = messages
        self._model = model
        self._cancelled = False

    def run(self) -> None:
        """Execute the completion request in background thread."""
        import asyncio

        log_constant(
            _LOGGER,
            LOG_UI_CHAT_REQUEST_STARTED,
            extra={"model": self._model.model_id, "provider": self._model.provider.value},
        )

        try:
            # Create new event loop for this thread
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

            try:
                result = loop.run_until_complete(
                    self._service.chat_completion(
                        messages=self._messages,
                        model=self._model,
                    )
                )
                if not self._cancelled:
                    log_constant(
                        _LOGGER,
                        LOG_UI_CHAT_REQUEST_COMPLETED,
                        extra={"model": self._model.model_id, "has_content": bool(result.content)},
                    )
                    self.finished.emit(result)
                else:
                    log_constant(_LOGGER, LOG_UI_CHAT_REQUEST_CANCELLED)
            finally:
                loop.close()

        except Exception as e:
            if not self._cancelled:
                log_constant(
                    _LOGGER,
                    LOG_UI_CHAT_REQUEST_ERROR,
                    message=f"Chat completion failed for {self._model.model_id}",
                    exc_info=e,
                )
                self.error.emit(str(e))

    def cancel(self) -> None:
        """Mark the request as cancelled."""
        self._cancelled = True
        log_constant(_LOGGER, LOG_UI_CHAT_REQUEST_CANCELLED)


class ChatPanel(QtWidgets.QWidget):
    """Chat panel for LLM interaction.

    Features:
    - API key input with show/hide toggle (stored in config, not .env)
    - Provider and model selection
    - Chat history display with incremental append
    - Message input with send button
    - Async request handling with cancellation support
    """

    # Signals
    api_key_changed = pyqtSignal(str)
    model_changed = pyqtSignal(object)  # ModelIdentity
    message_sent = pyqtSignal(str)

    def __init__(
        self,
        config: Optional[LLMConfig] = None,
        parent: Optional[QtWidgets.QWidget] = None,
    ) -> None:
        super().__init__(parent)

        self._config = config or LLMConfig.from_environment()
        self._service = LLMService(self._config)
        self._chat_history: List[ChatMessage] = []
        self._current_worker: Optional[ChatWorker] = None
        self._download_worker: Optional[ModelDownloadWorker] = None
        self._gpu_worker: Optional[GPUDetectionWorker] = None
        self._hf_validation_worker: Optional[HFTokenValidationWorker] = None
        self._hf_save_worker: Optional[HFTokenSaveWorker] = None
        self._is_busy = False
        self._model_manager = ModelManager.get_instance()
        self._gpu_result: Optional[GPUDetectionResult] = None

        self._setup_ui()
        self._connect_signals()
        self._populate_models()
        self._update_api_key_status()
        self._load_hf_token()
        self._start_gpu_detection()

    def _setup_ui(self) -> None:
        """Build the UI components."""
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(4)

        # Provider and Model selection row
        provider_row = QtWidgets.QHBoxLayout()
        provider_label = QtWidgets.QLabel("Provider:")
        self._provider_combo = QtWidgets.QComboBox()
        self._provider_combo.addItem("OpenRouter (Cloud)", LLMProvider.OPENROUTER)
        self._provider_combo.addItem("vLLM (Local)", LLMProvider.VLLM)
        self._provider_combo.setMaximumWidth(150)

        model_label = QtWidgets.QLabel("Model:")
        self._model_combo = QtWidgets.QComboBox()
        self._model_combo.setSizePolicy(
            QtWidgets.QSizePolicy.Policy.Expanding,
            QtWidgets.QSizePolicy.Policy.Fixed,
        )

        provider_row.addWidget(provider_label)
        provider_row.addWidget(self._provider_combo)
        provider_row.addWidget(model_label)
        provider_row.addWidget(self._model_combo, 1)
        layout.addLayout(provider_row)

        # API Key section (collapsible) - for OpenRouter
        self._api_key_group = QtWidgets.QGroupBox("API Configuration")
        self._api_key_group.setCheckable(True)
        self._api_key_group.setChecked(False)  # Start collapsed
        api_layout = QtWidgets.QVBoxLayout(self._api_key_group)
        api_layout.setContentsMargins(4, 4, 4, 4)
        api_layout.setSpacing(4)

        # API Key input row (OpenRouter)
        key_row = QtWidgets.QHBoxLayout()
        self._key_label = QtWidgets.QLabel("OpenRouter API Key:")
        self._api_key_input = QtWidgets.QLineEdit()
        self._api_key_input.setEchoMode(QtWidgets.QLineEdit.EchoMode.Password)
        self._api_key_input.setPlaceholderText("sk-or-v1-...")

        # Pre-fill from config or env
        existing_key = self._config.get_api_key()
        if existing_key and existing_key != "EMPTY":
            self._api_key_input.setText(existing_key)

        self._show_key_btn = QtWidgets.QPushButton("Show")
        self._show_key_btn.setCheckable(True)
        self._show_key_btn.setMaximumWidth(50)

        key_row.addWidget(self._key_label)
        key_row.addWidget(self._api_key_input, 1)
        key_row.addWidget(self._show_key_btn)
        api_layout.addLayout(key_row)

        # API Key status
        self._api_status_label = QtWidgets.QLabel("")
        self._api_status_label.setStyleSheet("font-size: 10px;")
        api_layout.addWidget(self._api_status_label)

        # Content widget for collapsible behavior
        self._api_content = QtWidgets.QWidget()
        self._api_content.setLayout(api_layout)
        api_group_layout = QtWidgets.QVBoxLayout(self._api_key_group)
        api_group_layout.setContentsMargins(0, 0, 0, 0)
        api_group_layout.addWidget(self._api_content)
        self._api_content.setVisible(False)

        layout.addWidget(self._api_key_group)

        # vLLM Settings section (collapsible) - contains HF, GPU, Proxy
        self._vllm_settings_group = QtWidgets.QGroupBox("vLLM Settings")
        self._vllm_settings_group.setCheckable(True)
        self._vllm_settings_group.setChecked(False)  # Start collapsed
        vllm_main_layout = QtWidgets.QVBoxLayout()
        vllm_main_layout.setContentsMargins(4, 4, 4, 4)
        vllm_main_layout.setSpacing(4)

        # --- HuggingFace Token subsection ---
        hf_header = QtWidgets.QLabel("HuggingFace Token")
        hf_header.setStyleSheet("font-weight: bold; font-size: 11px; color: #4a9eff;")
        vllm_main_layout.addWidget(hf_header)

        hf_row = QtWidgets.QHBoxLayout()
        self._hf_label = QtWidgets.QLabel("Token:")
        self._hf_token_input = QtWidgets.QLineEdit()
        self._hf_token_input.setEchoMode(QtWidgets.QLineEdit.EchoMode.Password)
        self._hf_token_input.setPlaceholderText("hf_...")

        self._show_hf_btn = QtWidgets.QPushButton("Show")
        self._show_hf_btn.setCheckable(True)
        self._show_hf_btn.setMaximumWidth(50)

        self._save_hf_btn = QtWidgets.QPushButton("Save")
        self._save_hf_btn.setMaximumWidth(50)

        self._validate_hf_btn = QtWidgets.QPushButton("Validate")
        self._validate_hf_btn.setMaximumWidth(60)

        hf_row.addWidget(self._hf_label)
        hf_row.addWidget(self._hf_token_input, 1)
        hf_row.addWidget(self._show_hf_btn)
        hf_row.addWidget(self._save_hf_btn)
        hf_row.addWidget(self._validate_hf_btn)
        vllm_main_layout.addLayout(hf_row)

        self._hf_status_label = QtWidgets.QLabel("")
        self._hf_status_label.setStyleSheet("font-size: 10px;")
        vllm_main_layout.addWidget(self._hf_status_label)

        self._hf_source_label = QtWidgets.QLabel("")
        self._hf_source_label.setStyleSheet("font-size: 9px; color: gray;")
        vllm_main_layout.addWidget(self._hf_source_label)

        # Model download progress
        self._download_progress = QtWidgets.QProgressBar()
        self._download_progress.setVisible(False)
        self._download_progress.setTextVisible(True)
        vllm_main_layout.addWidget(self._download_progress)

        # Separator
        sep1 = QtWidgets.QFrame()
        sep1.setFrameShape(QtWidgets.QFrame.Shape.HLine)
        sep1.setStyleSheet("color: #555;")
        vllm_main_layout.addWidget(sep1)

        # --- GPU Status subsection ---
        gpu_header_row = QtWidgets.QHBoxLayout()
        gpu_header = QtWidgets.QLabel("GPU Status")
        gpu_header.setStyleSheet("font-weight: bold; font-size: 11px; color: #4a9eff;")
        self._gpu_refresh_btn = QtWidgets.QPushButton("Refresh")
        self._gpu_refresh_btn.setMaximumWidth(60)
        gpu_header_row.addWidget(gpu_header)
        gpu_header_row.addStretch()
        gpu_header_row.addWidget(self._gpu_refresh_btn)
        vllm_main_layout.addLayout(gpu_header_row)

        self._gpu_status_label = QtWidgets.QLabel("Detecting GPUs...")
        self._gpu_status_label.setStyleSheet("font-size: 10px;")
        vllm_main_layout.addWidget(self._gpu_status_label)

        self._gpu_list = QtWidgets.QListWidget()
        self._gpu_list.setMaximumHeight(60)
        self._gpu_list.setStyleSheet("font-size: 10px;")
        vllm_main_layout.addWidget(self._gpu_list)

        self._gpu_recommendation_label = QtWidgets.QLabel("")
        self._gpu_recommendation_label.setStyleSheet("font-size: 9px; color: gray;")
        self._gpu_recommendation_label.setWordWrap(True)
        vllm_main_layout.addWidget(self._gpu_recommendation_label)

        # Separator
        sep2 = QtWidgets.QFrame()
        sep2.setFrameShape(QtWidgets.QFrame.Shape.HLine)
        sep2.setStyleSheet("color: #555;")
        vllm_main_layout.addWidget(sep2)

        # --- Proxy subsection ---
        proxy_header_row = QtWidgets.QHBoxLayout()
        proxy_header = QtWidgets.QLabel("Proxy")
        proxy_header.setStyleSheet("font-weight: bold; font-size: 11px; color: #4a9eff;")
        self._proxy_toggle = QtWidgets.QCheckBox("Enable")
        self._proxy_status_label = QtWidgets.QLabel("Disabled")
        self._proxy_status_label.setStyleSheet("font-size: 9px; color: gray;")
        proxy_header_row.addWidget(proxy_header)
        proxy_header_row.addWidget(self._proxy_toggle)
        proxy_header_row.addStretch()
        proxy_header_row.addWidget(self._proxy_status_label)
        vllm_main_layout.addLayout(proxy_header_row)

        proxy_inputs_row = QtWidgets.QHBoxLayout()
        self._http_proxy_input = QtWidgets.QLineEdit()
        self._http_proxy_input.setPlaceholderText("HTTP: http://127.0.0.1:7890")
        self._http_proxy_input.setEnabled(False)
        self._https_proxy_input = QtWidgets.QLineEdit()
        self._https_proxy_input.setPlaceholderText("HTTPS: https://127.0.0.1:7890")
        self._https_proxy_input.setEnabled(False)
        self._apply_proxy_btn = QtWidgets.QPushButton("Apply")
        self._apply_proxy_btn.setEnabled(False)
        self._apply_proxy_btn.setMaximumWidth(60)
        proxy_inputs_row.addWidget(self._http_proxy_input)
        proxy_inputs_row.addWidget(self._https_proxy_input)
        proxy_inputs_row.addWidget(self._apply_proxy_btn)
        vllm_main_layout.addLayout(proxy_inputs_row)

        # Content widget for collapsible behavior
        self._vllm_content = QtWidgets.QWidget()
        self._vllm_content.setLayout(vllm_main_layout)
        vllm_group_layout = QtWidgets.QVBoxLayout(self._vllm_settings_group)
        vllm_group_layout.setContentsMargins(0, 0, 0, 0)
        vllm_group_layout.addWidget(self._vllm_content)
        self._vllm_content.setVisible(False)

        self._vllm_settings_group.setVisible(False)  # Hidden by default
        layout.addWidget(self._vllm_settings_group)

        # Legacy group boxes (hidden, kept for compatibility)
        self._hf_token_group = QtWidgets.QWidget()
        self._hf_token_group.setVisible(False)
        self._gpu_group = QtWidgets.QWidget()
        self._gpu_group.setVisible(False)
        self._proxy_group = QtWidgets.QWidget()
        self._proxy_group.setVisible(False)

        # Chat history display
        self._chat_display = QtWidgets.QTextEdit()
        self._chat_display.setReadOnly(True)
        self._chat_display.setAcceptRichText(True)
        self._chat_display.setPlaceholderText(
            "Chat with MOSAIC Assistant...\n\n"
            "OpenRouter (Cloud): Enter API key from openrouter.ai/keys\n"
            "vLLM (Local): Select model and send a message to auto-download"
        )
        layout.addWidget(self._chat_display, 1)

        # Input row
        input_row = QtWidgets.QHBoxLayout()
        self._input_field = QtWidgets.QLineEdit()
        self._input_field.setPlaceholderText("Type your message...")
        self._send_btn = QtWidgets.QPushButton("Send")
        self._send_btn.setMaximumWidth(60)
        self._cancel_btn = QtWidgets.QPushButton("Cancel")
        self._cancel_btn.setMaximumWidth(60)
        self._cancel_btn.setVisible(False)
        input_row.addWidget(self._input_field)
        input_row.addWidget(self._send_btn)
        input_row.addWidget(self._cancel_btn)
        layout.addLayout(input_row)

        # Status row
        status_row = QtWidgets.QHBoxLayout()
        self._status_label = QtWidgets.QLabel("Ready")
        self._status_label.setStyleSheet("color: gray; font-size: 10px;")
        self._clear_btn = QtWidgets.QPushButton("Clear Chat")
        self._clear_btn.setMaximumWidth(70)
        self._clear_btn.setStyleSheet("font-size: 10px;")
        status_row.addWidget(self._status_label)
        status_row.addStretch()
        status_row.addWidget(self._clear_btn)
        layout.addLayout(status_row)

    def _connect_signals(self) -> None:
        """Connect widget signals."""
        self._provider_combo.currentIndexChanged.connect(self._on_provider_changed)
        self._api_key_input.textChanged.connect(self._on_api_key_changed)
        self._show_key_btn.toggled.connect(self._on_show_key_toggled)
        self._model_combo.currentIndexChanged.connect(self._on_model_changed)
        self._send_btn.clicked.connect(self._on_send_clicked)
        self._input_field.returnPressed.connect(self._on_send_clicked)
        self._cancel_btn.clicked.connect(self._on_cancel_clicked)
        self._clear_btn.clicked.connect(self._on_clear_clicked)
        # Collapsible group signals
        self._api_key_group.toggled.connect(self._on_api_group_toggled)
        self._vllm_settings_group.toggled.connect(self._on_vllm_group_toggled)
        # HuggingFace token signals
        self._show_hf_btn.toggled.connect(self._on_show_hf_toggled)
        self._save_hf_btn.clicked.connect(self._on_save_hf_token)
        self._validate_hf_btn.clicked.connect(self._on_validate_hf_token)
        # GPU signals
        self._gpu_refresh_btn.clicked.connect(self._start_gpu_detection)
        # Proxy signals
        self._proxy_toggle.toggled.connect(self._on_proxy_toggled)
        self._apply_proxy_btn.clicked.connect(self._on_apply_proxy)

    def _on_api_group_toggled(self, checked: bool) -> None:
        """Toggle API configuration visibility."""
        self._api_content.setVisible(checked)

    def _on_vllm_group_toggled(self, checked: bool) -> None:
        """Toggle vLLM settings visibility."""
        self._vllm_content.setVisible(checked)

    def _populate_models(self) -> None:
        """Populate model dropdown based on selected provider."""
        self._model_combo.clear()

        # Get current provider
        provider = self._provider_combo.currentData()
        if provider is None:
            provider = LLMProvider.OPENROUTER

        # Get models for this provider
        models = self._config.get_models_for_provider(provider)

        for model in models:
            self._model_combo.addItem(str(model), model)

        # Select first model
        if models:
            self._model_combo.setCurrentIndex(0)

    def _on_provider_changed(self, index: int) -> None:
        """Handle provider selection change."""
        provider = self._provider_combo.currentData()
        if provider is None:
            return

        # Update active provider in config
        self._config.active_provider = provider

        # Update models for new provider
        self._populate_models()

        # Update settings sections based on provider
        if provider == LLMProvider.OPENROUTER:
            self._key_label.setText("OpenRouter API Key:")
            self._api_key_input.setPlaceholderText("sk-or-v1-...")
            self._api_key_group.setVisible(True)
            self._vllm_settings_group.setVisible(False)
            # Load OpenRouter key
            existing_key = self._config.openrouter_api_key
            if existing_key:
                self._api_key_input.setText(existing_key)
            else:
                self._api_key_input.clear()
        elif provider == LLMProvider.VLLM:
            # vLLM doesn't require API key, hide API config
            self._api_key_group.setVisible(False)
            # Show consolidated vLLM settings
            self._vllm_settings_group.setVisible(True)
            self._update_hf_status()
            self._update_proxy_status()
            # Refresh GPU info if we have results
            if self._gpu_result:
                self._update_gpu_display()

        self._update_api_key_status()

    def _update_api_key_status(self) -> None:
        """Update API key status indicator."""
        provider = self._config.active_provider

        if provider == LLMProvider.VLLM:
            # vLLM doesn't require API key
            self._api_status_label.setText(
                f"vLLM server at {self._config.vllm_base_url}"
            )
            self._api_status_label.setStyleSheet("color: blue; font-size: 10px;")
            self._send_btn.setEnabled(True)
        elif self._config.has_valid_api_key():
            self._api_status_label.setText("API key configured")
            self._api_status_label.setStyleSheet("color: green; font-size: 10px;")
            self._send_btn.setEnabled(True)
        else:
            self._api_status_label.setText(
                "Enter your OpenRouter API key (get one at openrouter.ai/keys)"
            )
            self._api_status_label.setStyleSheet("color: orange; font-size: 10px;")
            self._send_btn.setEnabled(False)

    def _on_api_key_changed(self, text: str) -> None:
        """Handle API key input change."""
        self._config.set_api_key(text.strip())
        self._update_api_key_status()
        self.api_key_changed.emit(text.strip())

    def _on_show_key_toggled(self, checked: bool) -> None:
        """Toggle API key visibility."""
        if checked:
            self._api_key_input.setEchoMode(QtWidgets.QLineEdit.EchoMode.Normal)
            self._show_key_btn.setText("Hide")
        else:
            self._api_key_input.setEchoMode(QtWidgets.QLineEdit.EchoMode.Password)
            self._show_key_btn.setText("Show")

    def _on_show_hf_toggled(self, checked: bool) -> None:
        """Toggle HuggingFace token visibility."""
        if checked:
            self._hf_token_input.setEchoMode(QtWidgets.QLineEdit.EchoMode.Normal)
            self._show_hf_btn.setText("Hide")
        else:
            self._hf_token_input.setEchoMode(QtWidgets.QLineEdit.EchoMode.Password)
            self._show_hf_btn.setText("Show")

    def _load_hf_token(self) -> None:
        """Load existing HuggingFace token if available."""
        token = HuggingFaceAuth.get_token()
        if token:
            # Mask the token for display
            self._hf_token_input.setText(token)
            self._update_hf_status()

    def _on_save_hf_token(self) -> None:
        """Save the HuggingFace token using background worker to avoid GUI freeze."""
        token = self._hf_token_input.text().strip()
        if not token:
            self._hf_status_label.setText("Please enter a valid token")
            self._hf_status_label.setStyleSheet("color: orange; font-size: 10px;")
            return

        # Show saving status
        self._hf_status_label.setText("Saving token...")
        self._hf_status_label.setStyleSheet("color: blue; font-size: 10px;")
        self._save_hf_btn.setEnabled(False)

        # Start worker thread to save token (avoids GUI freeze from network calls)
        self._hf_save_worker = HFTokenSaveWorker(token, persist=True, parent=self)
        self._hf_save_worker.finished.connect(self._on_hf_save_finished)
        self._hf_save_worker.start()

    def _on_hf_save_finished(self, success: bool, message: str) -> None:
        """Handle HF token save result."""
        self._save_hf_btn.setEnabled(True)
        self._hf_save_worker = None

        if success:
            self._hf_status_label.setText(message)
            self._hf_status_label.setStyleSheet("color: green; font-size: 10px;")
            # Update the token source display
            self._update_hf_status()
        else:
            self._hf_status_label.setText(message)
            self._hf_status_label.setStyleSheet("color: red; font-size: 10px;")

    def _update_hf_status(self) -> None:
        """Update HuggingFace authentication status."""
        if HuggingFaceAuth.is_authenticated():
            self._hf_status_label.setText("HuggingFace authenticated")
            self._hf_status_label.setStyleSheet("color: green; font-size: 10px;")
            # Show token source
            source = HuggingFaceAuth.get_token_source()
            self._hf_source_label.setText(f"Token source: {source}")
        else:
            self._hf_status_label.setText(
                "Enter HuggingFace token for gated models (Llama, etc.) - "
                "Get token at huggingface.co/settings/tokens"
            )
            self._hf_status_label.setStyleSheet("color: orange; font-size: 10px;")
            self._hf_source_label.setText("")

    def _on_validate_hf_token(self) -> None:
        """Validate the HuggingFace token with HF API."""
        token = self._hf_token_input.text().strip()
        if not token:
            self._hf_status_label.setText("Please enter a token to validate")
            self._hf_status_label.setStyleSheet("color: orange; font-size: 10px;")
            return

        self._hf_status_label.setText("Validating token...")
        self._hf_status_label.setStyleSheet("color: blue; font-size: 10px;")
        self._validate_hf_btn.setEnabled(False)

        self._hf_validation_worker = HFTokenValidationWorker(token, parent=self)
        self._hf_validation_worker.finished.connect(self._on_hf_validation_finished)
        self._hf_validation_worker.start()

    def _on_hf_validation_finished(self, is_valid: bool, message: str) -> None:
        """Handle HF token validation result."""
        self._validate_hf_btn.setEnabled(True)
        self._hf_validation_worker = None

        if is_valid:
            self._hf_status_label.setText(message)
            self._hf_status_label.setStyleSheet("color: green; font-size: 10px;")
        else:
            self._hf_status_label.setText(message)
            self._hf_status_label.setStyleSheet("color: red; font-size: 10px;")

    def _start_gpu_detection(self) -> None:
        """Start GPU detection in background."""
        if self._gpu_worker and self._gpu_worker.isRunning():
            return  # Already running

        self._gpu_status_label.setText("Detecting GPUs...")
        self._gpu_list.clear()
        self._gpu_refresh_btn.setEnabled(False)

        self._gpu_worker = GPUDetectionWorker(parent=self)
        self._gpu_worker.finished.connect(self._on_gpu_detection_finished)
        self._gpu_worker.start()

    def _on_gpu_detection_finished(self, result: GPUDetectionResult) -> None:
        """Handle GPU detection result."""
        self._gpu_result = result
        self._gpu_worker = None
        self._gpu_refresh_btn.setEnabled(True)
        self._update_gpu_display()

    def _update_gpu_display(self) -> None:
        """Update the GPU display with detection results."""
        if not self._gpu_result:
            return

        result = self._gpu_result
        self._gpu_list.clear()

        if not result.has_gpus:
            if result.error_message:
                self._gpu_status_label.setText("GPU Detection Error")
                self._gpu_status_label.setStyleSheet("font-weight: bold; color: red;")
                self._gpu_list.addItem(f"Error: {result.error_message}")
            else:
                self._gpu_status_label.setText("No NVIDIA GPUs Detected")
                self._gpu_status_label.setStyleSheet("font-weight: bold; color: orange;")
                self._gpu_list.addItem("No NVIDIA GPUs found")
                self._gpu_list.addItem("Install NVIDIA drivers or use Cloud provider")

            self._gpu_recommendation_label.setText(
                "Consider using OpenRouter (Cloud) for LLM inference without a GPU"
            )
            return

        # Show GPU count and CUDA info
        gpu_count = len(result.gpus)
        cuda_info = f"CUDA {result.cuda_version}" if result.cuda_version else "CUDA N/A"
        self._gpu_status_label.setText(f"{gpu_count} GPU(s) Detected ({cuda_info})")
        self._gpu_status_label.setStyleSheet("font-weight: bold; color: green;")

        # List each GPU with VRAM info
        for gpu in result.gpus:
            # Create color-coded usage indicator
            usage = gpu.memory_usage_percent
            if usage < 50:
                color = "green"
            elif usage < 80:
                color = "orange"
            else:
                color = "red"

            item = QtWidgets.QListWidgetItem(
                f"GPU {gpu.index}: {gpu.name} | "
                f"{gpu.free_memory_gb:.1f}GB free / {gpu.total_memory_gb:.1f}GB total "
                f"({usage:.0f}% used)"
            )
            item.setForeground(QtGui.QColor(color))
            self._gpu_list.addItem(item)

        # Show model recommendations
        total_free = result.total_free_vram_gb
        recommendations = []
        if total_free >= 45:
            recommendations.append("Llama 3.1 70B")
        if total_free >= 10:
            recommendations.append("Llama 3.1 8B")
        if total_free >= 8:
            recommendations.append("Mistral 7B")
        if total_free >= 4:
            recommendations.append("Llama 3.2 3B")
        if total_free >= 2:
            recommendations.append("Llama 3.2 1B")

        if recommendations:
            self._gpu_recommendation_label.setText(
                f"Recommended models: {', '.join(recommendations[:3])}"
            )
        else:
            self._gpu_recommendation_label.setText(
                "Insufficient VRAM for local models. Consider using OpenRouter (Cloud)."
            )

    def _on_proxy_toggled(self, enabled: bool) -> None:
        """Handle proxy toggle change."""
        self._http_proxy_input.setEnabled(enabled)
        self._https_proxy_input.setEnabled(enabled)
        self._apply_proxy_btn.setEnabled(enabled)

        if not enabled:
            # Disable proxy immediately when toggled off
            ProxyConfig.set_proxy(False)
            log_constant(_LOGGER, LOG_UI_CHAT_PROXY_DISABLED)
            self._update_proxy_status()

    def _on_apply_proxy(self) -> None:
        """Apply proxy settings."""
        enabled = self._proxy_toggle.isChecked()
        http_proxy = self._http_proxy_input.text().strip()
        https_proxy = self._https_proxy_input.text().strip()

        ProxyConfig.set_proxy(enabled, http_proxy, https_proxy)

        if enabled:
            log_constant(
                _LOGGER,
                LOG_UI_CHAT_PROXY_ENABLED,
                extra={"http_proxy": http_proxy, "https_proxy": https_proxy},
            )
        else:
            log_constant(_LOGGER, LOG_UI_CHAT_PROXY_DISABLED)

        self._update_proxy_status()

    def _update_proxy_status(self) -> None:
        """Update proxy status display."""
        status = ProxyConfig.get_status()
        if ProxyConfig.is_enabled():
            self._proxy_status_label.setText(status)
            self._proxy_status_label.setStyleSheet("font-size: 10px; color: green;")
        else:
            self._proxy_status_label.setText(status)
            self._proxy_status_label.setStyleSheet("font-size: 10px; color: gray;")

    def _on_model_changed(self, index: int) -> None:
        """Handle model selection change."""
        if index >= 0:
            model = self._model_combo.itemData(index)
            if model:
                self.model_changed.emit(model)

    def _on_send_clicked(self) -> None:
        """Handle send button click."""
        text = self._input_field.text().strip()
        if not text:
            return

        # Check provider-specific requirements
        provider = self._config.active_provider

        if provider == LLMProvider.OPENROUTER:
            if not self._config.has_valid_api_key():
                self._show_error("Please enter your OpenRouter API key first.")
                return
        elif provider == LLMProvider.VLLM:
            # Check if model is ready (downloaded and server running)
            model = self._model_combo.currentData()
            if model and not self._check_vllm_model_ready(model):
                return  # Error/download already handled

        if self._is_busy:
            return

        self._input_field.clear()
        self.message_sent.emit(text)
        self._send_message_async(text)

    def _check_vllm_model_ready(self, model: ModelIdentity) -> bool:
        """Check if vLLM model is ready, trigger download if needed.

        Returns:
            True if model is ready, False if download needed or error.
        """
        model_info = self._model_manager.get_model_info(
            model.model_id, model.display_name or model.model_id
        )

        # Check if server is already running with this model
        if self._model_manager.server.is_running:
            if self._model_manager.server.current_model == model.model_id:
                return True
            # Different model - need to stop and restart
            self._status_label.setText("Switching models...")

        # Check authentication for gated models
        if model_info.requires_auth and not HuggingFaceAuth.is_authenticated():
            # Check if this is a Llama model (requires auth)
            if "llama" in model.model_id.lower():
                self._show_error(
                    "This model requires HuggingFace authentication.\n\n"
                    "1. Get a token at huggingface.co/settings/tokens\n"
                    "2. Accept the Llama license at huggingface.co/meta-llama\n"
                    "3. Enter and save your token above"
                )
                return False

        # Check if model is downloaded
        if model_info.status == ModelStatus.NOT_DOWNLOADED:
            self._start_model_download(model.model_id)
            return False
        elif model_info.status == ModelStatus.DOWNLOADING:
            self._show_error("Model is currently downloading. Please wait...")
            return False
        elif model_info.status == ModelStatus.ERROR:
            self._show_error(f"Model error: {model_info.error_message}")
            return False
        elif model_info.status in (ModelStatus.DOWNLOADED, ModelStatus.READY):
            # Start server if not running
            if not self._model_manager.server.is_running:
                self._start_model_download(model.model_id)  # This will start the server
                return False
            return True

        return False

    def _start_model_download(self, model_id: str) -> None:
        """Start downloading a model in the background."""
        if self._download_worker and self._download_worker.isRunning():
            self._show_error("A download is already in progress...")
            return

        self._download_progress.setVisible(True)
        self._download_progress.setValue(0)
        self._status_label.setText(f"Preparing {model_id}...")
        self._set_busy(True)

        self._download_worker = ModelDownloadWorker(model_id, parent=self)
        self._download_worker.progress.connect(self._on_download_progress)
        self._download_worker.finished.connect(self._on_download_finished)
        self._download_worker.error.connect(self._on_download_error)
        self._download_worker.start()

    def _on_download_progress(self, percentage: float, message: str) -> None:
        """Handle download progress updates."""
        self._download_progress.setValue(int(percentage))
        self._status_label.setText(message)

    def _on_download_finished(self, model_path: str) -> None:
        """Handle download completion."""
        self._download_progress.setVisible(False)
        self._set_busy(False)
        self._status_label.setText("Model ready! You can now send messages.")
        self._download_worker = None
        _LOGGER.info(f"Model ready at: {model_path}")

    def _on_download_error(self, error: str) -> None:
        """Handle download error."""
        self._download_progress.setVisible(False)
        self._set_busy(False)
        self._download_worker = None
        self._show_error(f"Download failed: {error}")

    def _on_cancel_clicked(self) -> None:
        """Handle cancel button click."""
        if self._current_worker and self._current_worker.isRunning():
            self._current_worker.cancel()
            self._current_worker.wait(1000)  # Wait up to 1 second
            self._set_busy(False)
            self._status_label.setText("Request cancelled")

    def _on_clear_clicked(self) -> None:
        """Handle clear button click."""
        self._chat_history.clear()
        self._chat_display.clear()
        self._status_label.setText("Chat cleared")

    def _send_message_async(self, content: str) -> None:
        """Send message and get response asynchronously using QThread."""
        # Add user message to history
        user_msg = ChatMessage(role="user", content=content)
        self._chat_history.append(user_msg)
        self._append_message("user", content)

        # Get selected model
        model = self._model_combo.currentData()
        if not model:
            self._show_error("No model selected")
            return

        # Update UI state
        self._set_busy(True)

        # Create worker thread
        self._current_worker = ChatWorker(
            service=self._service,
            messages=list(self._chat_history),  # Copy to avoid race conditions
            model=model,
            parent=self,
        )
        self._current_worker.finished.connect(self._on_completion_finished)
        self._current_worker.error.connect(self._on_completion_error)
        self._current_worker.start()

    def _on_completion_finished(self, result: CompletionResult) -> None:
        """Handle completion result from worker thread."""
        self._set_busy(False)
        self._current_worker = None

        if result.cancelled:
            self._status_label.setText("Request cancelled")
            return

        if result.error:
            self._show_error(result.error)
            return

        # Add assistant response to history
        assistant_msg = ChatMessage(role="assistant", content=result.content)
        self._chat_history.append(assistant_msg)
        self._append_message("assistant", result.content)

        # Update status with token info
        if result.tokens_used:
            self._status_label.setText(f"Ready ({result.tokens_used} tokens)")
        else:
            self._status_label.setText("Ready")

    def _on_completion_error(self, error: str) -> None:
        """Handle error from worker thread."""
        self._set_busy(False)
        self._current_worker = None
        _LOGGER.error(f"Chat completion failed: {error}")
        self._show_error(f"Error: {error}")

    def _set_busy(self, busy: bool) -> None:
        """Update UI busy state."""
        self._is_busy = busy
        self._input_field.setEnabled(not busy)
        self._send_btn.setEnabled(not busy and self._config.has_valid_api_key())
        self._send_btn.setVisible(not busy)
        self._cancel_btn.setVisible(busy)
        self._model_combo.setEnabled(not busy)
        self._api_key_input.setEnabled(not busy)

        if busy:
            self._status_label.setText("Thinking...")

    def _append_message(self, role: str, content: str) -> None:
        """Append a message to the chat display.

        Uses incremental append for performance (not full rebuild).
        """
        cursor = self._chat_display.textCursor()
        cursor.movePosition(QtGui.QTextCursor.MoveOperation.End)

        # Escape HTML and preserve newlines
        escaped = self._escape_html(content)

        # Format based on role with distinct visual styling
        if role == "user":
            # User message - right-aligned blue bubble style
            cursor.insertHtml(
                '<table width="100%" cellpadding="0" cellspacing="0" style="margin-bottom: 12px;">'
                '<tr><td align="right">'
                '<div style="background-color: #2563eb; color: white; padding: 8px 12px; '
                'border-radius: 12px 12px 0 12px; display: inline-block; max-width: 80%;">'
                f'<b style="color: #93c5fd;">You</b><br/>{escaped}'
                '</div></td></tr></table>'
            )
        else:
            # Assistant message - left-aligned green bubble style
            cursor.insertHtml(
                '<table width="100%" cellpadding="0" cellspacing="0" style="margin-bottom: 12px;">'
                '<tr><td align="left">'
                '<div style="background-color: #1e3a2f; color: #d1fae5; padding: 8px 12px; '
                'border-radius: 12px 12px 12px 0; display: inline-block; max-width: 80%;">'
                '<b style="color: #34d399;">MOSAIC Assistant</b><br/>'
                f'{escaped}'
                '</div></td></tr></table>'
            )

        # Scroll to bottom
        scrollbar = self._chat_display.verticalScrollBar()
        if scrollbar is not None:
            scrollbar.setValue(scrollbar.maximum())

    def _show_error(self, error: str) -> None:
        """Show error in chat display."""
        cursor = self._chat_display.textCursor()
        cursor.movePosition(QtGui.QTextCursor.MoveOperation.End)
        cursor.insertHtml(
            f'<p style="color: red; margin: 8px 0;"><b>Error:</b> {self._escape_html(error)}</p>'
        )
        scrollbar = self._chat_display.verticalScrollBar()
        if scrollbar is not None:
            scrollbar.setValue(scrollbar.maximum())
        self._status_label.setText("Error occurred")

    @staticmethod
    def _escape_html(text: str) -> str:
        """Escape HTML for safe display."""
        return (
            text.replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
            .replace("\n", "<br>")
        )

    def get_config(self) -> LLMConfig:
        """Get the current LLM config."""
        return self._config

    def set_api_key(self, api_key: str) -> None:
        """Programmatically set the API key."""
        self._api_key_input.setText(api_key)

    def cleanup(self) -> None:
        """Clean up resources."""
        if self._current_worker and self._current_worker.isRunning():
            self._current_worker.cancel()
            self._current_worker.wait(1000)
        if self._download_worker and self._download_worker.isRunning():
            self._download_worker.cancel()
            self._download_worker.wait(1000)
        if self._gpu_worker and self._gpu_worker.isRunning():
            self._gpu_worker.wait(1000)
        if self._hf_validation_worker and self._hf_validation_worker.isRunning():
            self._hf_validation_worker.wait(1000)
        if self._hf_save_worker and self._hf_save_worker.isRunning():
            self._hf_save_worker.wait(1000)
        # Shutdown vLLM server if running
        if self._model_manager:
            self._model_manager.shutdown()


__all__ = [
    "ChatPanel",
    "ChatWorker",
    "ModelDownloadWorker",
    "GPUDetectionWorker",
    "HFTokenValidationWorker",
    "HFTokenSaveWorker",
]
