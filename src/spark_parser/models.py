"""Immutable canonical models for parser configuration metadata."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from spark_parser.data_types import SparkDataType
from spark_parser.defaults import (
    DEFAULT_ARRAY_DISTINCT,
    DEFAULT_AUDIT,
    DEFAULT_BINARY_ENCODING,
    DEFAULT_BOOLEAN_CASE_SENSITIVE,
    DEFAULT_BOOLEAN_FALSE_VALUES,
    DEFAULT_BOOLEAN_TRUE_VALUES,
    DEFAULT_BOOLEAN_VALUES_MODE,
    DEFAULT_CHILD_ERROR_MODE,
    DEFAULT_COLLAPSE_WHITESPACE,
    DEFAULT_COMPLEX_INPUT_FORMAT,
    DEFAULT_DROP_NULL_ELEMENTS,
    DEFAULT_DROP_NULL_VALUES,
    DEFAULT_EMPTY_IS_NULL,
    DEFAULT_IS_NULLABLE,
    DEFAULT_NULL_MARKER_CASE_SENSITIVE,
    DEFAULT_NULL_MARKERS,
    DEFAULT_NULL_MARKERS_MODE,
    DEFAULT_ON_PARSE_ERROR,
    DEFAULT_REPLACE_NULL_MARKERS,
    DEFAULT_STRING_FORMAT,
    DEFAULT_TRIM_WHITESPACE,
    DEFAULT_ZERO_IS_VALID,
)
from spark_parser.enums import (
    BinaryEncoding,
    BooleanValuesMode,
    ChildErrorMode,
    ComplexInputFormat,
    NullMarkersMode,
    ParseErrorMode,
    ParserType,
    StringFormat,
)


@dataclass(frozen=True)
class ParserGlobals:
    """Options inherited by every configured column."""

    null_markers: tuple[str, ...] = DEFAULT_NULL_MARKERS
    null_marker_case_sensitive: bool = DEFAULT_NULL_MARKER_CASE_SENSITIVE
    true_values: tuple[str, ...] = DEFAULT_BOOLEAN_TRUE_VALUES
    false_values: tuple[str, ...] = DEFAULT_BOOLEAN_FALSE_VALUES
    boolean_case_sensitive: bool = DEFAULT_BOOLEAN_CASE_SENSITIVE


@dataclass(frozen=True)
class ParserOptions:
    """Fully resolved options for one parser implementation."""

    parser_type: ParserType
    trim_whitespace: bool = DEFAULT_TRIM_WHITESPACE
    collapse_whitespace: bool = DEFAULT_COLLAPSE_WHITESPACE
    empty_is_null: bool = DEFAULT_EMPTY_IS_NULL
    replace_null_markers: bool = DEFAULT_REPLACE_NULL_MARKERS
    null_markers: tuple[str, ...] = DEFAULT_NULL_MARKERS
    null_markers_mode: NullMarkersMode = DEFAULT_NULL_MARKERS_MODE
    null_marker_case_sensitive: bool = DEFAULT_NULL_MARKER_CASE_SENSITIVE
    is_nullable: bool = DEFAULT_IS_NULLABLE
    default_on_null: Any = None
    on_parse_error: ParseErrorMode = DEFAULT_ON_PARSE_ERROR
    default_on_error: Any = None
    audit: bool = DEFAULT_AUDIT
    zero_is_valid: bool = DEFAULT_ZERO_IS_VALID
    string_format: StringFormat | None = DEFAULT_STRING_FORMAT
    formats: tuple[str, ...] = ()
    true_values: tuple[str, ...] = DEFAULT_BOOLEAN_TRUE_VALUES
    false_values: tuple[str, ...] = DEFAULT_BOOLEAN_FALSE_VALUES
    boolean_case_sensitive: bool = DEFAULT_BOOLEAN_CASE_SENSITIVE
    boolean_values_mode: BooleanValuesMode = DEFAULT_BOOLEAN_VALUES_MODE
    binary_encoding: BinaryEncoding = DEFAULT_BINARY_ENCODING
    input_format: ComplexInputFormat = DEFAULT_COMPLEX_INPUT_FORMAT
    delimiter: str | None = None
    element_parser: NestedValueParser | None = None
    field_parsers: tuple[StructFieldParser, ...] = ()
    value_parser: NestedValueParser | None = None
    on_element_error: ChildErrorMode = DEFAULT_CHILD_ERROR_MODE
    on_value_error: ChildErrorMode = DEFAULT_CHILD_ERROR_MODE
    drop_null_elements: bool = DEFAULT_DROP_NULL_ELEMENTS
    distinct: bool = DEFAULT_ARRAY_DISTINCT
    drop_null_values: bool = DEFAULT_DROP_NULL_VALUES


@dataclass(frozen=True)
class NestedValueParser:
    """Parser bound to an array element or map value datatype."""

    expected_data_type: str
    data_type: SparkDataType
    parser: ParserOptions


@dataclass(frozen=True)
class StructFieldParser:
    """Source-to-silver parser for one configured struct field."""

    source_field_name: str
    silver_field_name: str
    expected_data_type: str
    data_type: SparkDataType
    parser: ParserOptions


@dataclass(frozen=True)
class ColumnParser:
    """One source column, target Spark datatype, and parser."""

    source_column_name: str
    silver_column_name: str
    expected_data_type: str
    data_type: SparkDataType
    parser: ParserOptions


@dataclass(frozen=True)
class ParserConfig:
    """One immutable, load-specific parsing configuration version."""

    parser_config_id: str
    parser_config_name: str
    version: str
    columns: tuple[ColumnParser, ...]
    globals: ParserGlobals = ParserGlobals()
    description: str | None = None
    owner: str | None = None
    owner_department: str | None = None
