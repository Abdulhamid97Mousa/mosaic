"""Qt-friendly validators and widgets for form validation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional, Tuple

from PyQt6 import QtCore, QtWidgets
from PyQt6.QtCore import pyqtSignal  # type: ignore[attr-defined]


@dataclass
class ValidationResult:
    """Result of a validation check."""

    is_valid: bool
    message: str = ""


class IntRangeValidator:
    """Validates integer input within a given range."""

    def __init__(self, min_val: int, max_val: int, field_name: str = "Value") -> None:
        self.min_val = min_val
        self.max_val = max_val
        self.field_name = field_name

    def validate(self, value: str) -> ValidationResult:
        if not value or not value.strip():
            return ValidationResult(False, f"{self.field_name} cannot be empty")

        try:
            num = int(value)
        except ValueError:
            return ValidationResult(False, f"{self.field_name} must be an integer")

        if num < self.min_val or num > self.max_val:
            return ValidationResult(
                False,
                f"{self.field_name} must be between {self.min_val} and {self.max_val}",
            )
        return ValidationResult(True)


class FloatRangeValidator:
    """Validates float input within a given range."""

    def __init__(
        self,
        min_val: float,
        max_val: float,
        field_name: str = "Value",
        inclusive_min: bool = True,
        inclusive_max: bool = True,
    ) -> None:
        self.min_val = min_val
        self.max_val = max_val
        self.field_name = field_name
        self.inclusive_min = inclusive_min
        self.inclusive_max = inclusive_max

    def validate(self, value: str) -> ValidationResult:
        if not value or not value.strip():
            return ValidationResult(False, f"{self.field_name} cannot be empty")

        try:
            num = float(value)
        except ValueError:
            return ValidationResult(False, f"{self.field_name} must be a number")

        if self.inclusive_min:
            if num < self.min_val:
                return ValidationResult(
                    False,
                    f"{self.field_name} must be >= {self.min_val}",
                )
        else:
            if num <= self.min_val:
                return ValidationResult(
                    False,
                    f"{self.field_name} must be > {self.min_val}",
                )

        if self.inclusive_max:
            if num > self.max_val:
                return ValidationResult(
                    False,
                    f"{self.field_name} must be <= {self.max_val}",
                )
        else:
            if num >= self.max_val:
                return ValidationResult(
                    False,
                    f"{self.field_name} must be < {self.max_val}",
                )

        return ValidationResult(True)


class NonEmptyStringValidator:
    """Ensures string values are not blank and satisfy optional regex patterns."""

    def __init__(
        self,
        field_name: str = "Value",
        *,
        min_length: int = 1,
        max_length: Optional[int] = None,
        pattern: Optional[str] = None,
    ) -> None:
        self.field_name = field_name
        self.min_length = min_length
        self.max_length = max_length
        self.pattern = pattern

    def validate(self, value: str) -> ValidationResult:
        if not value or not value.strip():
            return ValidationResult(False, f"{self.field_name} cannot be empty")

        stripped = value.strip()
        length = len(stripped)

        if length < self.min_length:
            return ValidationResult(
                False,
                f"{self.field_name} must be at least {self.min_length} characters",
            )

        if self.max_length is not None and length > self.max_length:
            return ValidationResult(
                False,
                f"{self.field_name} must be at most {self.max_length} characters",
            )

        if self.pattern is not None:
            import re

            if not re.match(self.pattern, stripped):
                return ValidationResult(False, f"{self.field_name} format is invalid")

        return ValidationResult(True)


class ValidatedLineEdit(QtWidgets.QLineEdit):
    """QLineEdit with integrated validation and red error display."""

    validation_changed = pyqtSignal(bool)  # True if valid, False if invalid

    def __init__(
        self,
        validator: Optional[Callable[[str], ValidationResult]] = None,
        parent: Optional[QtWidgets.QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self._validator = validator
        self._error_label = QtWidgets.QLabel(self.parentWidget())
        self._error_label.setStyleSheet("color: #ff4444; font-size: 10px; font-weight: bold;")
        self._error_label.setVisible(False)
        self._is_valid = True

        self.textChanged.connect(self._on_text_changed)

    def _on_text_changed(self, text: str) -> None:
        if self._validator is None:
            self._is_valid = True
            self._error_label.setVisible(False)
            self.validation_changed.emit(True)
            return

        result = self._validator(text)
        self._is_valid = result.is_valid

        if result.is_valid:
            self.setStyleSheet("")
            self._error_label.setVisible(False)
        else:
            self.setStyleSheet("border: 2px solid #ff4444;")
            self._error_label.setText(result.message)
            self._error_label.setVisible(True)

        self.validation_changed.emit(result.is_valid)

    def is_valid(self) -> bool:
        return self._is_valid

    def get_error_label(self) -> QtWidgets.QLabel:
        return self._error_label


class ValidatedSpinBox(QtWidgets.QSpinBox):
    """QSpinBox with integrated validation for ranges."""

    validation_changed = pyqtSignal(bool)

    def __init__(
        self,
        min_val: int,
        max_val: int,
        field_name: str = "Value",
        parent: Optional[QtWidgets.QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self.setRange(min_val, max_val)
        self._validator = IntRangeValidator(min_val, max_val, field_name)
        self._error_label = QtWidgets.QLabel(self.parentWidget())
        self._error_label.setStyleSheet("color: #ff4444; font-size: 10px; font-weight: bold;")
        self._error_label.setVisible(False)

        self.valueChanged.connect(self._on_value_changed)

    def _on_value_changed(self, value: int) -> None:
        result = self._validator.validate(str(value))
        if result.is_valid:
            self.setStyleSheet("")
            self._error_label.setVisible(False)
        else:
            self.setStyleSheet("border: 2px solid #ff4444;")
            self._error_label.setText(result.message)
            self._error_label.setVisible(True)

        self.validation_changed.emit(result.is_valid)

    def is_valid(self) -> bool:
        result = self._validator.validate(str(self.value()))
        return result.is_valid

    def get_error_label(self) -> QtWidgets.QLabel:
        return self._error_label


def create_validated_input_row(
    label_text: str,
    validator: Callable[[str], ValidationResult],
    parent: Optional[QtWidgets.QWidget] = None,
    *,
    placeholder: str = "",
    initial_value: str = "",
) -> Tuple[QtWidgets.QLabel, ValidatedLineEdit, QtWidgets.QLabel]:
    label = QtWidgets.QLabel(label_text, parent)
    input_widget = ValidatedLineEdit(validator, parent)
    input_widget.setPlaceholderText(placeholder)
    input_widget.setText(initial_value)
    error_label = input_widget.get_error_label()
    return label, input_widget, error_label


__all__ = [
    "ValidationResult",
    "IntRangeValidator",
    "FloatRangeValidator",
    "NonEmptyStringValidator",
    "ValidatedLineEdit",
    "ValidatedSpinBox",
    "create_validated_input_row",
]
