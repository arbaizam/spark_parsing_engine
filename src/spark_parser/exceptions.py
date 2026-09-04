"""Small public exception hierarchy for predictable caller error handling.

Compilation and schema binding fail before Spark execution where possible. Row-value failures may
still surface later from lazy Spark actions, while recoverable missing-source conditions use a
warning category rather than an exception.
"""

from __future__ import annotations

from collections.abc import Sequence

_NO_ERROR_ARGUMENT = object()


class SparkParserError(Exception):
    """Base exception for package failures, optionally containing several validation errors.

    A single error keeps its original string representation for backward compatibility. Validation
    phases that can safely inspect independent inputs may pass a sequence and expose every problem
    through :attr:`errors` while callers continue catching the same public exception classes.
    """

    def __init__(
        self,
        errors: object = _NO_ERROR_ARGUMENT,
        *legacy_args: object,
    ) -> None:
        """Store aggregate messages while retaining ordinary ``Exception`` construction."""
        if errors is _NO_ERROR_ARGUMENT:
            self.errors: tuple[str, ...] = ()
            super().__init__()
            return
        if legacy_args:
            super().__init__(errors, *legacy_args)
            rendered = str(self)
            self.errors = (rendered,) if rendered else ()
            return

        if isinstance(errors, str):
            normalized = (errors,) if errors else ()
        elif isinstance(errors, Sequence):
            candidates = tuple(errors)
            normalized = (
                candidates
                if candidates and all(isinstance(error, str) and error for error in candidates)
                else ()
            )
        else:
            normalized = ()
        if not normalized:
            super().__init__(errors)
            rendered = str(self)
            self.errors = (rendered,) if rendered else ()
            return

        self.errors = normalized
        message = (
            normalized[0]
            if len(normalized) == 1
            else f"Validation failed with {len(normalized)} errors:\n"
            + "\n".join(f"- {error}" for error in normalized)
        )
        super().__init__(message)


class CompilationError(SparkParserError):
    """Raised when parser YAML does not satisfy the authoring contract."""


class SchemaValidationError(SparkParserError):
    """Raised when an input DataFrame is incompatible with a parser config."""


class SchemaWarning(UserWarning):
    """Warn that parsing can continue with a recoverable schema mismatch."""
