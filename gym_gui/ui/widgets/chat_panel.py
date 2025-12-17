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

from gym_gui.services.llm import (
    LLMConfig,
    LLMProvider,
    LLMService,
    ModelIdentity,
    ChatMessage,
    CompletionResult,
)

if TYPE_CHECKING:
    pass

_LOGGER = logging.getLogger(__name__)


class ChatWorker(QtCore.QThread):
    """Worker thread for LLM chat completion requests.

    Runs HTTP requests in a background thread to avoid blocking Qt UI.
    """

    finished = QtCore.Signal(object)  # CompletionResult
    error = QtCore.Signal(str)

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
                    self.finished.emit(result)
            finally:
                loop.close()

        except Exception as e:
            if not self._cancelled:
                self.error.emit(str(e))

    def cancel(self) -> None:
        """Mark the request as cancelled."""
        self._cancelled = True


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
    api_key_changed = QtCore.Signal(str)
    model_changed = QtCore.Signal(object)  # ModelIdentity
    message_sent = QtCore.Signal(str)

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
        self._is_busy = False

        self._setup_ui()
        self._connect_signals()
        self._populate_models()
        self._update_api_key_status()

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

        # API Key section (collapsible)
        self._api_key_group = QtWidgets.QGroupBox("API Configuration")
        api_layout = QtWidgets.QVBoxLayout(self._api_key_group)
        api_layout.setContentsMargins(4, 4, 4, 4)
        api_layout.setSpacing(4)

        # API Key input row
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

        layout.addWidget(self._api_key_group)

        # Chat history display
        self._chat_display = QtWidgets.QTextEdit()
        self._chat_display.setReadOnly(True)
        self._chat_display.setAcceptRichText(True)
        self._chat_display.setPlaceholderText(
            "Chat with MOSAIC Assistant...\n\n"
            "OpenRouter (Cloud): Enter API key from openrouter.ai/keys\n"
            "vLLM (Local): Start server with 'vllm serve <model>'"
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

        # Update API key section based on provider
        if provider == LLMProvider.OPENROUTER:
            self._key_label.setText("OpenRouter API Key:")
            self._api_key_input.setPlaceholderText("sk-or-v1-...")
            self._api_key_group.setVisible(True)
            # Load OpenRouter key
            existing_key = self._config.openrouter_api_key
            if existing_key:
                self._api_key_input.setText(existing_key)
            else:
                self._api_key_input.clear()
        elif provider == LLMProvider.VLLM:
            self._key_label.setText("vLLM API Key (optional):")
            self._api_key_input.setPlaceholderText("EMPTY (default)")
            # vLLM doesn't require API key, hide or minimize
            self._api_key_group.setVisible(False)

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

        if not self._config.has_valid_api_key():
            self._show_error("Please enter your OpenRouter API key first.")
            return

        if self._is_busy:
            return

        self._input_field.clear()
        self.message_sent.emit(text)
        self._send_message_async(text)

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

        # Format based on role
        if role == "user":
            cursor.insertHtml('<p style="color: #0066cc; margin: 8px 0;"><b>You:</b></p>')
        else:
            cursor.insertHtml(
                '<p style="color: #006600; margin: 8px 0;"><b>Assistant:</b></p>'
            )

        # Escape HTML and preserve newlines
        escaped = self._escape_html(content)
        cursor.insertHtml(f'<p style="margin: 0 0 12px 0;">{escaped}</p>')

        # Scroll to bottom
        self._chat_display.verticalScrollBar().setValue(
            self._chat_display.verticalScrollBar().maximum()
        )

    def _show_error(self, error: str) -> None:
        """Show error in chat display."""
        cursor = self._chat_display.textCursor()
        cursor.movePosition(QtGui.QTextCursor.MoveOperation.End)
        cursor.insertHtml(
            f'<p style="color: red; margin: 8px 0;"><b>Error:</b> {self._escape_html(error)}</p>'
        )
        self._chat_display.verticalScrollBar().setValue(
            self._chat_display.verticalScrollBar().maximum()
        )
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


__all__ = ["ChatPanel", "ChatWorker"]
