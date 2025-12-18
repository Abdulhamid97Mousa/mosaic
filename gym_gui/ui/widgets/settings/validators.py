"""Validators for settings dialog input fields.

This module provides validation functions for different types of settings values.
These are used by both the SettingsService and UI widgets to provide real-time
validation feedback.
"""

from __future__ import annotations

import re
from typing import List, Optional, Tuple
from urllib.parse import urlparse


class SettingValidators:
    """Collection of static validation methods for setting values."""

    @staticmethod
    def validate_boolean(value: str) -> Tuple[bool, str]:
        """Validate boolean value.

        Accepts: 0, 1, true, false, yes, no, on, off (case-insensitive)

        Args:
            value: Value to validate

        Returns:
            Tuple of (is_valid, error_message)
        """
        valid_values = {"0", "1", "true", "false", "yes", "no", "on", "off"}
        if value.lower() in valid_values:
            return True, ""
        return False, f"Must be one of: {', '.join(sorted(valid_values))}"

    @staticmethod
    def validate_integer(
        value: str,
        min_val: Optional[int] = None,
        max_val: Optional[int] = None,
    ) -> Tuple[bool, str]:
        """Validate integer value with optional range check.

        Args:
            value: Value to validate
            min_val: Optional minimum value (inclusive)
            max_val: Optional maximum value (inclusive)

        Returns:
            Tuple of (is_valid, error_message)
        """
        try:
            int_value = int(value)

            if min_val is not None and int_value < min_val:
                return False, f"Must be at least {min_val}"

            if max_val is not None and int_value > max_val:
                return False, f"Must be at most {max_val}"

            return True, ""
        except ValueError:
            return False, "Must be a valid integer"

    @staticmethod
    def validate_url(value: str) -> Tuple[bool, str]:
        """Validate URL format.

        Empty values are considered valid (for optional proxy settings).

        Args:
            value: URL to validate

        Returns:
            Tuple of (is_valid, error_message)
        """
        if not value:
            return True, ""  # Empty is okay (optional)

        try:
            result = urlparse(value)
            # URL must have both scheme (http/https) and netloc (hostname)
            if not all([result.scheme, result.netloc]):
                return False, "Must be a valid URL (e.g., http://example.com)"

            # Check scheme is http or https
            if result.scheme not in ("http", "https"):
                return False, "URL scheme must be http or https"

            return True, ""
        except Exception as e:
            return False, f"Invalid URL format: {str(e)}"

    @staticmethod
    def validate_email(value: str) -> Tuple[bool, str]:
        """Validate email address format.

        Empty values are considered valid (for optional settings).

        Args:
            value: Email address to validate

        Returns:
            Tuple of (is_valid, error_message)
        """
        if not value:
            return True, ""  # Empty is okay (optional)

        # RFC 5322 simplified email regex
        # Matches most common email formats but not all edge cases
        email_regex = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'

        if re.match(email_regex, value):
            return True, ""

        return False, "Must be a valid email address (e.g., user@example.com)"

    @staticmethod
    def validate_enum(value: str, options: List[str]) -> Tuple[bool, str]:
        """Validate that value is in the list of allowed options.

        Args:
            value: Value to validate
            options: List of valid options

        Returns:
            Tuple of (is_valid, error_message)
        """
        if not options:
            # No options specified, accept any value
            return True, ""

        if value in options:
            return True, ""

        return False, f"Must be one of: {', '.join(options)}"

    @staticmethod
    def validate_string(
        value: str,
        min_length: Optional[int] = None,
        max_length: Optional[int] = None,
        pattern: Optional[str] = None,
    ) -> Tuple[bool, str]:
        """Validate string value with optional constraints.

        Args:
            value: String to validate
            min_length: Optional minimum length
            max_length: Optional maximum length
            pattern: Optional regex pattern to match

        Returns:
            Tuple of (is_valid, error_message)
        """
        if min_length is not None and len(value) < min_length:
            return False, f"Must be at least {min_length} characters long"

        if max_length is not None and len(value) > max_length:
            return False, f"Must be at most {max_length} characters long"

        if pattern is not None:
            try:
                if not re.match(pattern, value):
                    return False, f"Must match pattern: {pattern}"
            except re.error as e:
                return False, f"Invalid regex pattern: {str(e)}"

        return True, ""


__all__ = ["SettingValidators"]
