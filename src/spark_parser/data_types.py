"""Spark datatype parsing independent of a running Spark session."""

from __future__ import annotations

import re
from dataclasses import dataclass

from spark_parser.enums import COMPLEX_PARSER_TYPES, ParserType
from spark_parser.exceptions import CompilationError

_IDENTIFIER = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
_TYPE_ALIASES = {
    "tinyint": "byte",
    "smallint": "short",
    "int": "integer",
    "bigint": "long",
    "real": "float",
    "bool": "boolean",
    "timestamp_ltz": "timestamp",
}


@dataclass(frozen=True)
class SparkStructField:
    """One field in a parsed Spark struct type."""

    name: str
    data_type: SparkDataType


@dataclass(frozen=True)
class SparkDataType:
    """Canonical recursive Spark datatype description."""

    parser_type: ParserType
    precision: int | None = None
    scale: int | None = None
    element_type: SparkDataType | None = None
    key_type: SparkDataType | None = None
    value_type: SparkDataType | None = None
    fields: tuple[SparkStructField, ...] = ()

    @property
    def is_complex(self) -> bool:
        return self.parser_type in COMPLEX_PARSER_TYPES

    @property
    def supports_equality(self) -> bool:
        """Whether Spark can compare values for operations such as array_distinct."""
        if self.parser_type is ParserType.MAP:
            return False
        if self.parser_type is ParserType.ARRAY:
            assert self.element_type is not None
            return self.element_type.supports_equality
        if self.parser_type is ParserType.STRUCT:
            return all(field.data_type.supports_equality for field in self.fields)
        return True

    @property
    def canonical(self) -> str:
        if self.parser_type is ParserType.DECIMAL:
            return f"decimal({self.precision},{self.scale})"
        if self.parser_type is ParserType.ARRAY:
            assert self.element_type is not None
            return f"array<{self.element_type.canonical}>"
        if self.parser_type is ParserType.MAP:
            assert self.key_type is not None and self.value_type is not None
            return f"map<{self.key_type.canonical},{self.value_type.canonical}>"
        if self.parser_type is ParserType.STRUCT:
            rendered = ",".join(
                f"{_quote_field(field.name)}:{field.data_type.canonical}" for field in self.fields
            )
            return f"struct<{rendered}>"
        return self.parser_type.value


def canonical_type_name(value: str) -> str:
    """Return the canonical spelling of one Spark datatype or parser alias."""
    normalized = value.strip().lower()
    return _TYPE_ALIASES.get(normalized, normalized)


def parse_spark_data_type(value: str) -> SparkDataType:
    """Parse and canonicalize the supported Spark SQL datatype grammar."""
    if not isinstance(value, str) or not value.strip():
        raise CompilationError("expected_data_type must be a non-empty string.")
    parser = _DataTypeParser(value)
    parsed = parser.parse_type()
    parser.skip_whitespace()
    if not parser.at_end:
        parser.fail("Unexpected trailing datatype content")
    return parsed


def _quote_field(name: str) -> str:
    if _IDENTIFIER.fullmatch(name):
        return name
    return f"`{name.replace('`', '``')}`"


class _DataTypeParser:
    def __init__(self, text: str) -> None:
        self.text = text
        self.position = 0

    @property
    def at_end(self) -> bool:
        return self.position >= len(self.text)

    def skip_whitespace(self) -> None:
        while not self.at_end and self.text[self.position].isspace():
            self.position += 1

    def parse_type(self) -> SparkDataType:
        type_name = canonical_type_name(self.read_identifier("datatype"))
        if type_name in {"decimal", "dec", "numeric"}:
            return self.parse_decimal()
        if type_name == "array":
            self.expect("<")
            element_type = self.parse_type()
            self.expect(">")
            return SparkDataType(ParserType.ARRAY, element_type=element_type)
        if type_name == "map":
            self.expect("<")
            key_type = self.parse_type()
            self.expect(",")
            value_type = self.parse_type()
            self.expect(">")
            return SparkDataType(ParserType.MAP, key_type=key_type, value_type=value_type)
        if type_name == "struct":
            return self.parse_struct()
        try:
            parser_type = ParserType(type_name)
        except ValueError as exc:
            supported = ", ".join(member.value for member in ParserType)
            self.fail(
                f"Unsupported datatype {type_name!r}; supported types are {supported} "
                "and decimal(p,s)",
                cause=exc,
            )
        if parser_type in COMPLEX_PARSER_TYPES or parser_type is ParserType.DECIMAL:
            self.fail(f"Datatype {type_name!r} requires its complete type parameters")
        return SparkDataType(parser_type)

    def parse_decimal(self) -> SparkDataType:
        self.expect("(")
        precision = self.read_integer("decimal precision")
        self.expect(",")
        scale = self.read_integer("decimal scale")
        self.expect(")")
        if not 1 <= precision <= 38:
            raise CompilationError("Decimal precision must be between 1 and 38.")
        if not 0 <= scale <= precision:
            raise CompilationError("Decimal scale must be between 0 and its precision.")
        return SparkDataType(ParserType.DECIMAL, precision=precision, scale=scale)

    def parse_struct(self) -> SparkDataType:
        self.expect("<")
        fields: list[SparkStructField] = []
        self.skip_whitespace()
        if self.peek(">"):
            self.fail("Struct types must contain at least one field")
        while True:
            field_name = self.read_field_name()
            self.expect(":")
            fields.append(SparkStructField(field_name, self.parse_type()))
            self.skip_whitespace()
            if self.peek(">"):
                self.position += 1
                break
            self.expect(",")
        names = [field.name for field in fields]
        duplicates = sorted({name for name in names if names.count(name) > 1})
        if duplicates:
            raise CompilationError(f"Struct datatype contains duplicate fields: {duplicates}.")
        return SparkDataType(ParserType.STRUCT, fields=tuple(fields))

    def read_field_name(self) -> str:
        self.skip_whitespace()
        if self.peek("`"):
            self.position += 1
            pieces: list[str] = []
            while not self.at_end:
                char = self.text[self.position]
                if char != "`":
                    pieces.append(char)
                    self.position += 1
                    continue
                if self.position + 1 < len(self.text) and self.text[self.position + 1] == "`":
                    pieces.append("`")
                    self.position += 2
                    continue
                self.position += 1
                if not pieces:
                    self.fail("Struct field names may not be empty")
                return "".join(pieces)
            self.fail("Unterminated backtick-quoted struct field name")
        return self.read_identifier("struct field name", preserve_case=True)

    def read_identifier(self, label: str, *, preserve_case: bool = False) -> str:
        self.skip_whitespace()
        match = _IDENTIFIER.match(self.text, self.position)
        if match is None:
            self.fail(f"Expected {label}")
        self.position = match.end()
        value = match.group(0)
        return value if preserve_case else value.lower()

    def read_integer(self, label: str) -> int:
        self.skip_whitespace()
        start = self.position
        while not self.at_end and self.text[self.position].isdigit():
            self.position += 1
        if start == self.position:
            self.fail(f"Expected {label}")
        return int(self.text[start : self.position])

    def expect(self, token: str) -> None:
        self.skip_whitespace()
        if not self.peek(token):
            self.fail(f"Expected {token!r}")
        self.position += len(token)

    def peek(self, token: str) -> bool:
        return self.text.startswith(token, self.position)

    def fail(self, message: str, *, cause: Exception | None = None):
        error = CompilationError(
            f"Invalid expected_data_type at character {self.position + 1}: {message}. "
            f"Value: {self.text!r}."
        )
        if cause is not None:
            raise error from cause
        raise error
