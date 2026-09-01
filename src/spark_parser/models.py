"""Immutable canonical models shared by compilation, execution, and reporting.

These dataclasses contain fully resolved values, not raw YAML. That distinction is important: code
after compilation never needs to guess whether an option was omitted, inherited, or defaulted.
Frozen instances also make configuration hashing and lazy Spark-plan construction predictable.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from spark_parser.data_types import SparkDataType
from spark_parser.defaults import (
    DEFAULT_AUDIT,
    DEFAULT_BINARY_ENCODING,
    DEFAULT_BOOLEAN_CASE_SENSITIVE,
    DEFAULT_BOOLEAN_FALSE_VALUES,
    DEFAULT_BOOLEAN_TRUE_VALUES,
    DEFAULT_BOOLEAN_VALUES_MODE,
    DEFAULT_COLLAPSE_WHITESPACE,
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
    NullMarkersMode,
    ParseErrorMode,
    ParserType,
    StringFormat,
)


@dataclass(frozen=True)
class ParserGlobals:
    """Code-owned defaults and author overrides inherited by configured columns.

    Tuples are used instead of lists so the compiled global vocabulary cannot be mutated after
    individual parser options have inherited it.
    """

    null_markers: tuple[str, ...] = DEFAULT_NULL_MARKERS
    null_marker_case_sensitive: bool = DEFAULT_NULL_MARKER_CASE_SENSITIVE
    true_values: tuple[str, ...] = DEFAULT_BOOLEAN_TRUE_VALUES
    false_values: tuple[str, ...] = DEFAULT_BOOLEAN_FALSE_VALUES
    boolean_case_sensitive: bool = DEFAULT_BOOLEAN_CASE_SENSITIVE


@dataclass(frozen=True)
class ParserOptions:
    """Fully resolved options for one scalar parser."""

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


@dataclass(frozen=True)
class ColumnParser:
    """Top-level binding between one bronze source and one typed target output."""

    source_column_name: str
    target_column_name: str
    expected_data_type: str
    data_type: SparkDataType
    parser: ParserOptions


@dataclass(frozen=True)
class ParserConfig:
    """One immutable, load-specific parsing contract.

    Column order is meaningful: ``parsed_df`` preserves it, report tables display it, and audit
    entries follow it. Consumers should use ``parser_config_id``, ``version``, and the serializer's
    content hash together when recording lineage.
    """

    parser_config_id: str
    parser_config_name: str
    version: str
    columns: tuple[ColumnParser, ...]
    globals: ParserGlobals = ParserGlobals()
    description: str | None = None
    owner: str | None = None
    owner_department: str | None = None


def needs_spark_boolean_overlap_check(
    true_values: tuple[str, ...],
    false_values: tuple[str, ...],
    case_sensitive: bool,
) -> bool:
    """Return whether only Spark can decide case-insensitive vocabulary overlap exactly.

    Python and the JVM can ship different Unicode tables. Exact and ASCII-only comparisons are
    stable at compile time; a case-insensitive set containing non-ASCII text must be lowered and
    compared by the same Spark runtime that will parse the values.
    """
    return not case_sensitive and any(
        not value.isascii() for value in (*true_values, *false_values)
    )
