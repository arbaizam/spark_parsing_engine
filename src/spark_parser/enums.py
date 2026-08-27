"""Canonical configuration vocabulary."""

from enum import Enum


class ParserType(str, Enum):
    """Supported bronze-string parser implementations."""

    STRING = "string"
    INTEGER = "integer"
    LONG = "long"
    DECIMAL = "decimal"
    DOUBLE = "double"
    BOOLEAN = "boolean"
    DATE = "date"
    TIMESTAMP = "timestamp"


class NullMarkersMode(str, Enum):
    """How column null markers relate to global markers."""

    REPLACE = "replace"
    EXTEND = "extend"


class ParseErrorMode(str, Enum):
    """Allowed outcomes for a non-null value that cannot be parsed."""

    FAIL = "fail"
    NULL = "null"
    DEFAULT = "default"


class StringFormat(str, Enum):
    """Optional string case formatting."""

    LOWER = "lower"
    UPPER = "upper"
    PASCAL = "pascal"


NUMERIC_PARSER_TYPES = frozenset(
    {
        ParserType.INTEGER,
        ParserType.LONG,
        ParserType.DECIMAL,
        ParserType.DOUBLE,
    }
)
