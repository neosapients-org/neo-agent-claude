"""Custom exceptions for Neo Memory Hub."""

from typing import Any


class NeoMemoryError(Exception):
    """Base exception for all Neo Memory Hub errors."""

    def __init__(
        self,
        message: str,
        details: dict[str, Any] | None = None,
    ) -> None:
        self.message = message
        self.details = details or {}
        super().__init__(self.message)

    def __str__(self) -> str:
        if self.details:
            return f"{self.message} | Details: {self.details}"
        return self.message


class StorageError(NeoMemoryError):
    """Error during storage operations (store, retrieve, update, delete)."""

    pass


class ValidationError(NeoMemoryError):
    """Error during data validation."""

    def __init__(
        self,
        message: str,
        field: str | None = None,
        value: Any = None,
        details: dict[str, Any] | None = None,
    ) -> None:
        details = details or {}
        if field:
            details["field"] = field
        if value is not None:
            details["value"] = str(value)[:100]  # Truncate for safety
        super().__init__(message, details)


class IsolationError(NeoMemoryError):
    """Error related to tenant/user/agent isolation."""

    def __init__(
        self,
        message: str,
        scope: dict[str, str | None] | None = None,
        details: dict[str, Any] | None = None,
    ) -> None:
        details = details or {}
        if scope:
            details["scope"] = scope
        super().__init__(message, details)


class ConfigurationError(NeoMemoryError):
    """Error in configuration loading or validation."""

    pass


class ProviderError(NeoMemoryError):
    """Error from a storage or LLM provider."""

    def __init__(
        self,
        message: str,
        provider: str,
        operation: str | None = None,
        details: dict[str, Any] | None = None,
    ) -> None:
        details = details or {}
        details["provider"] = provider
        if operation:
            details["operation"] = operation
        super().__init__(message, details)


class NotFoundError(NeoMemoryError):
    """Resource not found error."""

    def __init__(
        self,
        resource_type: str,
        resource_id: str,
        details: dict[str, Any] | None = None,
    ) -> None:
        details = details or {}
        details["resource_type"] = resource_type
        details["resource_id"] = resource_id
        super().__init__(f"{resource_type} not found: {resource_id}", details)


class AccessDeniedError(NeoMemoryError):
    """Access denied error for isolation violations."""

    def __init__(
        self,
        message: str,
        requester_scope: dict[str, str | None] | None = None,
        target_scope: dict[str, str | None] | None = None,
        details: dict[str, Any] | None = None,
    ) -> None:
        details = details or {}
        if requester_scope:
            details["requester_scope"] = requester_scope
        if target_scope:
            details["target_scope"] = target_scope
        super().__init__(message, details)
