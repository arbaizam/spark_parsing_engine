"""Validate and canonicalize the scalar Spark datatypes supported by the package."""

from __future__ import annotations

import re
from dataclasses import dataclass

from spark_parser.enums import ParserType
from spark_parser.exceptions import CompilationError

_IDENTIFIER = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
_DECIMAL = re.compile(
    r"(?P<name>[A-Za-z_][A-Za-z0-9_]*)\s*\(\s*"
    r"(?P<precision>[0-9]+)\s*,\s*(?P<scale>[0-9]+)\s*\)"
)

# Common Spark SQL spellings accepted in both expected_data_type and parser declarations.
_TYPE_ALIASES = {
    "tinyint": "byte",
    "smallint": "short",
    "int": "integer",
    "bigint": "long",
    "real": "float",
    "bool": "boolean",
    "dec": "decimal",
    "numeric": "decimal",
    "timestamp_ltz": "timestamp",
}


@dataclass(frozen=True)
class SparkDataType:
    """Canonical description of one supported scalar Spark datatype."""

    parser_type: ParserType
    precision: int | None = None
    scale: int | None = None

    @property
    def canonical(self) -> str:
        """Render the canonical Spark DDL spelling."""
        if self.parser_type is ParserType.DECIMAL:
            return f"decimal({self.precision},{self.scale})"
        return self.parser_type.value


def canonical_type_name(value: str) -> str:
    """Return the canonical spelling of one scalar datatype or parser alias."""
    normalized = value.strip().lower()
    return _TYPE_ALIASES.get(normalized, normalized)


def parse_spark_data_type(value: str) -> SparkDataType:
    """Validate a scalar Spark datatype and return its canonical description."""
    if not isinstance(value, str) or not value.strip():
        raise CompilationError("expected_data_type must be a non-empty string.")
    try:
        value.encode("utf-8")
    except UnicodeEncodeError as exc:
        raise CompilationError(
            "expected_data_type must contain well-formed Unicode; "
            f"invalid code point at character {exc.start + 1}."
        ) from exc

    text = value.strip()
    decimal_match = _DECIMAL.fullmatch(text)
    if decimal_match:
        type_name = canonical_type_name(decimal_match.group("name"))
        if type_name != ParserType.DECIMAL.value:
            raise _unsupported_datatype(text)
        precision = _bounded_integer(decimal_match.group("precision"))
        scale = _bounded_integer(decimal_match.group("scale"))
        if not 1 <= precision <= 38:
            raise CompilationError("Decimal precision must be between 1 and 38.")
        if not 0 <= scale <= precision:
            raise CompilationError("Decimal scale must be between 0 and its precision.")
        return SparkDataType(ParserType.DECIMAL, precision=precision, scale=scale)

    if not _IDENTIFIER.fullmatch(text):
        if canonical_type_name(text.split("(", 1)[0]) == ParserType.DECIMAL.value:
            raise CompilationError(
                "Expected decimal precision and scale as ASCII integers in decimal(p,s)."
            )
        raise _unsupported_datatype(text)

    type_name = canonical_type_name(text)
    if type_name == ParserType.DECIMAL.value:
        raise CompilationError("expected_data_type decimal requires precision and scale.")
    try:
        parser_type = ParserType(type_name)
    except ValueError as exc:
        raise _unsupported_datatype(text) from exc
    return SparkDataType(parser_type)


def _bounded_integer(value: str) -> int:
    """Parse a decimal parameter without handing an adversarial integer to ``int``."""
    significant = value.lstrip("0") or "0"
    return 39 if len(significant) > 2 else int(significant)


def _unsupported_datatype(value: str) -> CompilationError:
    """Build the consistent public error for an unsupported datatype."""
    supported = ", ".join(member.value for member in ParserType if member is not ParserType.DECIMAL)
    return CompilationError(
        f"Unsupported datatype {value!r}; supported types are {supported}, and decimal(p,s)."
    )
