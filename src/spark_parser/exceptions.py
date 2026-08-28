"""Small public exception hierarchy for predictable caller error handling.

Compilation and schema binding fail before Spark execution where possible. Row-value failures may
still surface later from lazy Spark actions, while recoverable missing-source conditions use a
warning category rather than an exception.
"""


class SparkParserError(Exception):
    """Base exception for package failures."""


class CompilationError(SparkParserError):
    """Raised when parser YAML does not satisfy the authoring contract."""


class SchemaValidationError(SparkParserError):
    """Raised when an input DataFrame is incompatible with a parser config."""


class SchemaWarning(UserWarning):
    """Warn that parsing can continue with a recoverable schema mismatch."""
