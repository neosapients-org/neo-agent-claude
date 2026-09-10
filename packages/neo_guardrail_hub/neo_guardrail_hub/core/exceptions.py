"""Custom exceptions for Neo Guardrail Hub.

This module defines all custom exception types used throughout
the framework for specific error handling.
"""

from typing import Any, Dict, List, Optional


class GuardrailError(Exception):
    """Base exception for all guardrail-related errors.

    All other custom exceptions in this module inherit from this class,
    allowing for broad exception catching when needed.
    """

    def __init__(
        self,
        message: str,
        guardrail_name: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Initialize the exception.

        Args:
            message: Human-readable error message
            guardrail_name: Name of the guardrail that raised the error
            details: Additional error details
        """
        super().__init__(message)
        self.message = message
        self.guardrail_name = guardrail_name
        self.details = details or {}

    def __str__(self) -> str:
        if self.guardrail_name:
            return f"[{self.guardrail_name}] {self.message}"
        return self.message


class ConfigurationError(GuardrailError):
    """Raised when there's an error in configuration.

    This includes invalid YAML syntax, missing required fields,
    or invalid configuration values.
    """

    def __init__(
        self,
        message: str,
        config_path: Optional[str] = None,
        field: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Initialize the exception.

        Args:
            message: Human-readable error message
            config_path: Path to the configuration file with the error
            field: Specific field that has the error
            details: Additional error details
        """
        super().__init__(message, details=details)
        self.config_path = config_path
        self.field = field

    def __str__(self) -> str:
        parts = []
        if self.config_path:
            parts.append(f"in {self.config_path}")
        if self.field:
            parts.append(f"field '{self.field}'")
        if parts:
            return f"Configuration error {' '.join(parts)}: {self.message}"
        return f"Configuration error: {self.message}"


class ProviderError(GuardrailError):
    """Raised when there's an error with a guardrail provider.

    This includes provider initialization failures, unsupported
    guardrail types, or provider-specific errors.
    """

    def __init__(
        self,
        message: str,
        provider_name: Optional[str] = None,
        guardrail_type: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Initialize the exception.

        Args:
            message: Human-readable error message
            provider_name: Name of the provider that raised the error
            guardrail_type: Type of guardrail being requested
            details: Additional error details
        """
        super().__init__(message, details=details)
        self.provider_name = provider_name
        self.guardrail_type = guardrail_type

    def __str__(self) -> str:
        parts = []
        if self.provider_name:
            parts.append(f"provider '{self.provider_name}'")
        if self.guardrail_type:
            parts.append(f"guardrail '{self.guardrail_type}'")
        if parts:
            return f"Provider error ({', '.join(parts)}): {self.message}"
        return f"Provider error: {self.message}"


class ExecutionTimeoutError(GuardrailError):
    """Raised when a guardrail check exceeds the timeout limit.

    This is raised when parallel or sequential execution takes
    longer than the configured timeout.
    """

    def __init__(
        self,
        message: str,
        timeout_ms: int,
        guardrail_names: Optional[List[str]] = None,
        details: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Initialize the exception.

        Args:
            message: Human-readable error message
            timeout_ms: The timeout that was exceeded in milliseconds
            guardrail_names: Names of guardrails that were running
            details: Additional error details
        """
        super().__init__(message, details=details)
        self.timeout_ms = timeout_ms
        self.guardrail_names = guardrail_names or []

    def __str__(self) -> str:
        if self.guardrail_names:
            names = ", ".join(self.guardrail_names)
            return f"Execution timeout ({self.timeout_ms}ms) for guardrails [{names}]: {self.message}"
        return f"Execution timeout ({self.timeout_ms}ms): {self.message}"


class ValidationError(GuardrailError):
    """Raised when input validation fails.

    This is raised when the input to a guardrail or the framework
    doesn't meet the expected format or constraints.
    """

    def __init__(
        self,
        message: str,
        field: Optional[str] = None,
        value: Optional[Any] = None,
        expected: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Initialize the exception.

        Args:
            message: Human-readable error message
            field: Name of the field that failed validation
            value: The invalid value that was provided
            expected: Description of what was expected
            details: Additional error details
        """
        super().__init__(message, details=details)
        self.field = field
        self.value = value
        self.expected = expected

    def __str__(self) -> str:
        parts = [f"Validation error: {self.message}"]
        if self.field:
            parts.append(f"Field: {self.field}")
        if self.expected:
            parts.append(f"Expected: {self.expected}")
        return " | ".join(parts)


class GuardrailNotFoundError(GuardrailError):
    """Raised when a requested guardrail type is not found.

    This is raised when trying to get a guardrail that hasn't
    been registered with the registry.
    """

    def __init__(
        self,
        guardrail_type: str,
        available_types: Optional[List[str]] = None,
        details: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Initialize the exception.

        Args:
            guardrail_type: The guardrail type that was not found
            available_types: List of available guardrail types
            details: Additional error details
        """
        message = f"Guardrail type '{guardrail_type}' not found"
        super().__init__(message, guardrail_name=guardrail_type, details=details)
        self.guardrail_type = guardrail_type
        self.available_types = available_types or []

    def __str__(self) -> str:
        msg = f"Guardrail type '{self.guardrail_type}' not found"
        if self.available_types:
            available = ", ".join(self.available_types[:5])
            if len(self.available_types) > 5:
                available += f"... ({len(self.available_types)} total)"
            msg += f". Available types: {available}"
        return msg
