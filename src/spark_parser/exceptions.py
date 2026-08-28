"""Public exception hierarchy for :mod:`spark_parser`."""


class SparkParserError(Exception):
    """Base exception for package failures."""


class CompilationError(SparkParserError):
    """Raised when parser YAML does not satisfy the authoring contract."""


class SchemaValidationError(SparkParserError):
    """Raised when an input DataFrame is incompatible with a parser config."""


class SchemaWarning(UserWarning):
    """Warn that parsing can continue with a recoverable schema mismatch."""
