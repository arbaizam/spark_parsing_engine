"""Immutable canonical models for parser configuration metadata."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from spark_parser.enums import NullMarkersMode, ParseErrorMode, ParserType, StringFormat


@dataclass(frozen=True)
class ParserGlobals:
    """Options inherited by every configured column."""

    null_markers: tuple[str, ...] = ()
    null_marker_case_sensitive: bool = True


@dataclass(frozen=True)
class ParserOptions:
    """Fully resolved options for one parser implementation."""

    parser_type: ParserType
    trim_whitespace: bool = True
    collapse_whitespace: bool = True
    empty_is_null: bool = True
    replace_null_markers: bool = False
    null_markers: tuple[str, ...] = ()
    null_markers_mode: NullMarkersMode = NullMarkersMode.REPLACE
    null_marker_case_sensitive: bool = True
    is_nullable: bool = True
    default_on_null: Any = None
    on_parse_error: ParseErrorMode = ParseErrorMode.FAIL
    default_on_error: Any = None
    audit: bool = False
    zero_is_valid: bool = True
    string_format: StringFormat | None = None
    formats: tuple[str, ...] = ()
    true_values: tuple[str, ...] = ("true",)
    false_values: tuple[str, ...] = ("false",)
    boolean_case_sensitive: bool = False


@dataclass(frozen=True)
class ColumnParser:
    """One source column, target Spark datatype, and parser."""

    column_name: str
    data_type: str
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
