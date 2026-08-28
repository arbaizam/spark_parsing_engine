"""Canonical, closed vocabulary accepted by configuration and runtime code.

Enums prevent spelling drift between YAML compilation, Spark expression generation, metadata,
and audit output. Each value is also a string so serialized output stays straightforward.
"""

from enum import Enum


class ParserType(str, Enum):
    """Supported bronze-string parser implementations."""

    STRING = "string"
    BYTE = "byte"
    SHORT = "short"
    INTEGER = "integer"
    LONG = "long"
    FLOAT = "float"
    DECIMAL = "decimal"
    DOUBLE = "double"
    BINARY = "binary"
    BOOLEAN = "boolean"
    DATE = "date"
    TIMESTAMP = "timestamp"
    TIMESTAMP_NTZ = "timestamp_ntz"
    ARRAY = "array"
    STRUCT = "struct"
    MAP = "map"


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
    PRESERVE = "preserve"


class StringFormat(str, Enum):
    """Optional deterministic string formatting profile."""

    LOWER = "lower"
    UPPER = "upper"
    TITLE = "title"
    PASCAL = "pascal"
    ADDRESS_US_V1 = "address_us_v1"
    COUNTY = "county"
    STATE_US = "state_us"
    ZIP = "zip"


class ComplexInputFormat(str, Enum):
    """Supported bronze encodings for complex values."""

    JSON = "json"
    DELIMITED = "delimited"


class ChildErrorMode(str, Enum):
    """Resolution for an invalid array element or map value."""

    FAIL = "fail"
    NULL = "null"
    DROP = "drop"
    PRESERVE = "preserve"


class BinaryEncoding(str, Enum):
    """Supported encodings for bronze binary strings."""

    BASE64 = "base64"
    HEX = "hex"
    UTF8 = "utf8"


# Shared parser families keep conditional behavior centralized. ``frozenset`` documents that
# these groupings are constants and makes accidental runtime mutation impossible.
NUMERIC_PARSER_TYPES = frozenset(
    {
        ParserType.BYTE,
        ParserType.SHORT,
        ParserType.INTEGER,
        ParserType.LONG,
        ParserType.FLOAT,
        ParserType.DECIMAL,
        ParserType.DOUBLE,
    }
)

COMPLEX_PARSER_TYPES = frozenset(
    {
        ParserType.ARRAY,
        ParserType.STRUCT,
        ParserType.MAP,
    }
)
