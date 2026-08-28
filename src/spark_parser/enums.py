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


class BooleanValuesMode(str, Enum):
    """How column Boolean tokens relate to global tokens."""

    REPLACE = "replace"
    EXTEND = "extend"


class ParseErrorMode(str, Enum):
    """Allowed outcomes for a non-null value that cannot be parsed."""

    FAIL = "fail"
    NULL = "null"
    DEFAULT = "default"


class StringFormat(str, Enum):
    """Optional deterministic string formatting profile."""

    LOWER = "lower"
    UPPER = "upper"
    PASCAL = "pascal"
    ADDRESS_US_V1 = "address_us_v1"
    COUNTY = "county"
    ZIP = "zip"


NUMERIC_PARSER_TYPES = frozenset(
    {
        ParserType.INTEGER,
        ParserType.LONG,
        ParserType.DECIMAL,
        ParserType.DOUBLE,
    }
)
