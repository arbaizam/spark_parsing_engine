"""Strict YAML compiler for parser configuration metadata."""

from __future__ import annotations

import math
import re
from collections.abc import Mapping
from datetime import date, datetime
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any

import yaml

from spark_parser.enums import (
    NUMERIC_PARSER_TYPES,
    NullMarkersMode,
    ParseErrorMode,
    ParserType,
    StringFormat,
)
from spark_parser.exceptions import CompilationError
from spark_parser.models import ColumnParser, ParserConfig, ParserGlobals, ParserOptions

_MISSING = object()
_DECIMAL_PATTERN = re.compile(r"decimal\((\d+),(\d+)\)", re.IGNORECASE)
_TYPE_ALIASES = {
    "int": "integer",
    "bigint": "long",
}
_INTEGER_MIN = -(2**31)
_INTEGER_MAX = 2**31 - 1
_LONG_MIN = -(2**63)
_LONG_MAX = 2**63 - 1


class _UniqueKeyLoader(yaml.SafeLoader):
    """Safe YAML loader that rejects duplicate mapping keys."""


def _construct_unique_mapping(
    loader: _UniqueKeyLoader,
    node: yaml.MappingNode,
    deep: bool = False,
) -> dict[Any, Any]:
    mapping: dict[Any, Any] = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        if key in mapping:
            raise CompilationError(f"Duplicate YAML key: {key!r}.")
        mapping[key] = loader.construct_object(value_node, deep=deep)
    return mapping


_UniqueKeyLoader.add_constructor(
    yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG,
    _construct_unique_mapping,
)


class YamlParserConfigCompiler:
    """Compile strict YAML into a fully resolved :class:`ParserConfig`."""

    def compile_path(self, path: str | Path) -> ParserConfig:
        """Compile a UTF-8 YAML file."""
        try:
            text = Path(path).read_text(encoding="utf-8")
        except OSError as exc:
            raise CompilationError(f"Unable to read parser config {path!s}: {exc}") from exc
        return self.compile_text(text)

    def compile_text(self, text: str) -> ParserConfig:
        """Compile one YAML document from text."""
        try:
            payload = yaml.load(text, Loader=_UniqueKeyLoader)
        except CompilationError:
            raise
        except yaml.YAMLError as exc:
            raise CompilationError(f"Invalid parser YAML: {exc}") from exc
        return self.compile_mapping(self._ensure_mapping(payload, "parser config"))

    def compile_mapping(self, payload: Mapping[str, Any]) -> ParserConfig:
        """Compile a parsed YAML mapping."""
        self._reject_keys(
            payload,
            {
                "parser_config_id",
                "parser_config_name",
                "version",
                "description",
                "owner",
                "owner_department",
                "globals",
                "columns",
            },
            "Parser config",
        )
        globals_config = self._compile_globals(payload.get("globals", {}))
        raw_columns = payload.get("columns")
        if not isinstance(raw_columns, list) or not raw_columns:
            raise CompilationError("columns must be a non-empty list.")
        columns = tuple(
            self._compile_column(raw_column, globals_config, index)
            for index, raw_column in enumerate(raw_columns, start=1)
        )
        self._validate_unique_columns(columns)
        return ParserConfig(
            parser_config_id=self._required_string(payload, "parser_config_id"),
            parser_config_name=self._required_string(payload, "parser_config_name"),
            version=self._required_string(payload, "version"),
            columns=columns,
            globals=globals_config,
            description=self._optional_string(payload, "description"),
            owner=self._optional_string(payload, "owner"),
            owner_department=self._optional_string(payload, "owner_department"),
        )

    def _compile_globals(self, raw_globals: Any) -> ParserGlobals:
        payload = self._ensure_mapping(raw_globals, "globals")
        self._reject_keys(
            payload,
            {"null_markers", "null_marker_case_sensitive"},
            "globals",
        )
        return ParserGlobals(
            null_markers=self._string_sequence(payload.get("null_markers", []), "null_markers"),
            null_marker_case_sensitive=self._bool(
                payload,
                "null_marker_case_sensitive",
                True,
            ),
        )

    def _compile_column(
        self,
        raw_column: Any,
        globals_config: ParserGlobals,
        index: int,
    ) -> ColumnParser:
        payload = self._ensure_mapping(raw_column, f"column at index {index}")
        self._reject_keys(
            payload,
            {"column_name", "data_type", "parser"},
            f"Column at index {index}",
        )
        column_name = self._required_string(payload, "column_name")
        data_type, expected_parser_type = self._normalize_data_type(
            self._required_string(payload, "data_type")
        )
        options = self._compile_parser(
            payload.get("parser", _MISSING),
            globals_config,
            expected_parser_type,
            data_type,
            column_name,
        )
        return ColumnParser(
            column_name=column_name,
            data_type=data_type,
            parser=options,
        )

    def _compile_parser(
        self,
        raw_parser: Any,
        globals_config: ParserGlobals,
        expected_type: ParserType,
        data_type: str,
        column_name: str,
    ) -> ParserOptions:
        if raw_parser is _MISSING:
            raise CompilationError(f"parser is required for column {column_name!r}.")
        if isinstance(raw_parser, str):
            payload: Mapping[str, Any] = {"type": raw_parser}
        else:
            payload = self._ensure_mapping(raw_parser, f"parser for {column_name!r}")

        parser_type = self._parser_type(self._required_string(payload, "type"))
        if parser_type is not expected_type:
            raise CompilationError(
                f"Parser {parser_type.value!r} is incompatible with data_type {data_type!r} "
                f"for column {column_name!r}; expected {expected_type.value!r}."
            )
        allowed_keys = {
            "type",
            "trim_whitespace",
            "replace_null_markers",
            "null_markers",
            "null_markers_mode",
            "null_marker_case_sensitive",
            "is_nullable",
            "default_on_null",
            "on_parse_error",
            "default_on_error",
            "audit",
        }
        if parser_type in NUMERIC_PARSER_TYPES:
            allowed_keys.add("zero_is_valid")
        if parser_type is ParserType.STRING:
            allowed_keys.add("format")
        if parser_type in {ParserType.DATE, ParserType.TIMESTAMP}:
            allowed_keys.add("formats")
        if parser_type is ParserType.BOOLEAN:
            allowed_keys.update({"true_values", "false_values", "boolean_case_sensitive"})
        self._reject_keys(payload, allowed_keys, f"Parser for {column_name!r}")

        markers_mode = self._enum_value(
            NullMarkersMode,
            payload.get("null_markers_mode", NullMarkersMode.REPLACE.value),
            "null_markers_mode",
        )
        if "null_markers_mode" in payload and "null_markers" not in payload:
            raise CompilationError(
                f"null_markers_mode for {column_name!r} requires column null_markers."
            )
        if "null_markers" in payload:
            column_markers = self._string_sequence(payload["null_markers"], "null_markers")
            markers = (
                self._deduplicate((*globals_config.null_markers, *column_markers))
                if markers_mode is NullMarkersMode.EXTEND
                else column_markers
            )
        else:
            markers = globals_config.null_markers
        replace_null_markers = self._bool(payload, "replace_null_markers", False)
        if replace_null_markers and not markers:
            raise CompilationError(
                f"replace_null_markers is true for {column_name!r}, but no null markers exist."
            )

        is_nullable = self._bool(payload, "is_nullable", True)
        raw_default_on_null = payload.get("default_on_null", _MISSING)
        if not is_nullable and raw_default_on_null is _MISSING:
            raise CompilationError(
                f"Column {column_name!r} is not nullable and requires default_on_null."
            )
        if is_nullable and raw_default_on_null is not _MISSING:
            raise CompilationError(
                f"default_on_null for {column_name!r} requires is_nullable: false."
            )
        default_on_null = (
            self._typed_default(raw_default_on_null, parser_type, data_type, "default_on_null")
            if raw_default_on_null is not _MISSING
            else None
        )

        raw_on_parse_error = payload.get("on_parse_error", ParseErrorMode.FAIL.value)
        # YAML resolves an unquoted ``null`` scalar to ``None``. In this one
        # enum position it unambiguously names the canonical ``null`` mode.
        if "on_parse_error" in payload and raw_on_parse_error is None:
            raw_on_parse_error = ParseErrorMode.NULL.value
        on_parse_error = self._enum_value(
            ParseErrorMode,
            raw_on_parse_error,
            "on_parse_error",
        )
        raw_default_on_error = payload.get("default_on_error", _MISSING)
        if on_parse_error is ParseErrorMode.DEFAULT and raw_default_on_error is _MISSING:
            raise CompilationError(
                f"on_parse_error: default for {column_name!r} requires default_on_error."
            )
        if on_parse_error is not ParseErrorMode.DEFAULT and raw_default_on_error is not _MISSING:
            raise CompilationError(
                f"default_on_error for {column_name!r} requires on_parse_error: default."
            )
        default_on_error = (
            self._typed_default(raw_default_on_error, parser_type, data_type, "default_on_error")
            if raw_default_on_error is not _MISSING
            else None
        )

        zero_is_valid = self._bool(payload, "zero_is_valid", True)
        if (
            parser_type in NUMERIC_PARSER_TYPES
            and not zero_is_valid
            and default_on_null is not None
            and Decimal(str(default_on_null)) == 0
        ):
            raise CompilationError(
                f"Column {column_name!r} rejects zero but uses zero as default_on_null."
            )

        string_format = self._compile_string_format(payload, parser_type)
        formats = self._compile_formats(payload, parser_type)
        true_values, false_values, boolean_case_sensitive = self._compile_boolean_values(
            payload,
            parser_type,
            column_name,
        )
        return ParserOptions(
            parser_type=parser_type,
            trim_whitespace=self._bool(payload, "trim_whitespace", True),
            replace_null_markers=replace_null_markers,
            null_markers=markers,
            null_markers_mode=markers_mode,
            null_marker_case_sensitive=self._bool(
                payload,
                "null_marker_case_sensitive",
                globals_config.null_marker_case_sensitive,
            ),
            is_nullable=is_nullable,
            default_on_null=default_on_null,
            on_parse_error=on_parse_error,
            default_on_error=default_on_error,
            audit=self._bool(payload, "audit", False),
            zero_is_valid=zero_is_valid,
            string_format=string_format,
            formats=formats,
            true_values=true_values,
            false_values=false_values,
            boolean_case_sensitive=boolean_case_sensitive,
        )

    def _compile_string_format(
        self,
        payload: Mapping[str, Any],
        parser_type: ParserType,
    ) -> StringFormat | None:
        if parser_type is not ParserType.STRING:
            return None
        value = payload.get("format")
        if value is None or value == "none":
            return None
        return self._enum_value(StringFormat, value, "format")

    def _compile_formats(
        self,
        payload: Mapping[str, Any],
        parser_type: ParserType,
    ) -> tuple[str, ...]:
        if parser_type is ParserType.DATE:
            default = ("yyyy-MM-dd",)
        elif parser_type is ParserType.TIMESTAMP:
            default = ("yyyy-MM-dd HH:mm:ss",)
        else:
            return ()
        if "formats" not in payload:
            return default
        formats = self._string_sequence(payload["formats"], "formats", allow_empty_values=False)
        if not formats:
            raise CompilationError("formats must contain at least one Spark datetime pattern.")
        return formats

    def _compile_boolean_values(
        self,
        payload: Mapping[str, Any],
        parser_type: ParserType,
        column_name: str,
    ) -> tuple[tuple[str, ...], tuple[str, ...], bool]:
        if parser_type is not ParserType.BOOLEAN:
            return ("true",), ("false",), False
        true_values = self._string_sequence(
            payload.get("true_values", ["true"]),
            "true_values",
            allow_empty_values=False,
        )
        false_values = self._string_sequence(
            payload.get("false_values", ["false"]),
            "false_values",
            allow_empty_values=False,
        )
        if not true_values or not false_values:
            raise CompilationError("true_values and false_values must be non-empty.")
        case_sensitive = self._bool(payload, "boolean_case_sensitive", False)
        normalize = (lambda item: item) if case_sensitive else (lambda item: item.casefold())
        overlap = {normalize(item) for item in true_values} & {
            normalize(item) for item in false_values
        }
        if overlap:
            raise CompilationError(
                f"Boolean true_values and false_values overlap for {column_name!r}: "
                f"{sorted(overlap)}."
            )
        return true_values, false_values, case_sensitive

    def _normalize_data_type(self, value: str) -> tuple[str, ParserType]:
        normalized = _TYPE_ALIASES.get(value.strip().lower(), value.strip().lower())
        decimal_match = _DECIMAL_PATTERN.fullmatch(normalized)
        if decimal_match:
            precision = int(decimal_match.group(1))
            scale = int(decimal_match.group(2))
            if not 1 <= precision <= 38:
                raise CompilationError("Decimal precision must be between 1 and 38.")
            if not 0 <= scale <= precision:
                raise CompilationError("Decimal scale must be between 0 and its precision.")
            return f"decimal({precision},{scale})", ParserType.DECIMAL
        try:
            parser_type = ParserType(normalized)
        except ValueError as exc:
            valid = ", ".join(member.value for member in ParserType)
            raise CompilationError(
                f"Unsupported data_type {value!r}. Valid types: {valid}, decimal(p,s)."
            ) from exc
        return parser_type.value, parser_type

    def _parser_type(self, value: str) -> ParserType:
        normalized = _TYPE_ALIASES.get(value.strip().lower(), value.strip().lower())
        try:
            return ParserType(normalized)
        except ValueError as exc:
            valid = ", ".join(member.value for member in ParserType)
            raise CompilationError(f"Invalid parser type {value!r}. Valid types: {valid}.") from exc

    def _typed_default(
        self,
        value: Any,
        parser_type: ParserType,
        data_type: str,
        label: str,
    ) -> Any:
        if value is None:
            raise CompilationError(f"{label} must be non-null.")
        if parser_type is ParserType.STRING:
            if not isinstance(value, str):
                raise CompilationError(f"{label} for string must be a string.")
            return value
        if parser_type in {ParserType.INTEGER, ParserType.LONG}:
            return self._integer_default(value, parser_type, label)
        if parser_type is ParserType.DOUBLE:
            return self._double_default(value, label)
        if parser_type is ParserType.DECIMAL:
            return self._decimal_default(value, data_type, label)
        if parser_type is ParserType.BOOLEAN:
            if not isinstance(value, bool):
                raise CompilationError(f"{label} for boolean must be true or false.")
            return value
        if parser_type is ParserType.DATE:
            return self._date_default(value, label)
        if parser_type is ParserType.TIMESTAMP:
            return self._timestamp_default(value, label)
        raise CompilationError(f"Unsupported parser type for {label}: {parser_type.value}.")

    @staticmethod
    def _integer_default(value: Any, parser_type: ParserType, label: str) -> int:
        if isinstance(value, bool) or not isinstance(value, int):
            raise CompilationError(f"{label} for {parser_type.value} must be an integer.")
        minimum, maximum = (
            (_INTEGER_MIN, _INTEGER_MAX)
            if parser_type is ParserType.INTEGER
            else (_LONG_MIN, _LONG_MAX)
        )
        if not minimum <= value <= maximum:
            raise CompilationError(f"{label} does not fit {parser_type.value}.")
        return value

    @staticmethod
    def _double_default(value: Any, label: str) -> float:
        if isinstance(value, bool) or not isinstance(value, (int, float, Decimal)):
            raise CompilationError(f"{label} for double must be numeric.")
        converted = float(value)
        if not math.isfinite(converted):
            raise CompilationError(f"{label} for double must be finite.")
        return converted

    @staticmethod
    def _decimal_default(value: Any, data_type: str, label: str) -> Decimal:
        try:
            converted = Decimal(str(value))
        except (InvalidOperation, ValueError) as exc:
            raise CompilationError(f"{label} for decimal must be numeric.") from exc
        if not converted.is_finite():
            raise CompilationError(f"{label} for decimal must be finite.")
        decimal_match = _DECIMAL_PATTERN.fullmatch(data_type)
        if decimal_match is None:
            raise CompilationError(f"Invalid canonical decimal type: {data_type}.")
        precision, scale = (int(part) for part in decimal_match.groups())
        _, digits, exponent = converted.as_tuple()
        integral_digits = max(len(digits) + exponent, 0)
        fractional_digits = max(-exponent, 0)
        if integral_digits > precision - scale or fractional_digits > scale:
            raise CompilationError(f"{label} does not fit {data_type}.")
        return converted

    @staticmethod
    def _date_default(value: Any, label: str) -> date:
        if isinstance(value, datetime):
            raise CompilationError(f"{label} for date must not contain a time component.")
        if isinstance(value, date):
            return value
        if isinstance(value, str):
            try:
                return date.fromisoformat(value)
            except ValueError as exc:
                raise CompilationError(f"{label} for date must use ISO YYYY-MM-DD.") from exc
        raise CompilationError(f"{label} for date must be a date or ISO string.")

    @staticmethod
    def _timestamp_default(value: Any, label: str) -> datetime:
        if isinstance(value, datetime):
            return value
        if isinstance(value, str):
            try:
                return datetime.fromisoformat(value)
            except ValueError as exc:
                raise CompilationError(f"{label} for timestamp must be an ISO timestamp.") from exc
        raise CompilationError(f"{label} for timestamp must be a datetime or ISO string.")

    def _validate_unique_columns(self, columns: tuple[ColumnParser, ...]) -> None:
        column_names = [column.column_name for column in columns]
        duplicate_columns = sorted({name for name in column_names if column_names.count(name) > 1})
        if duplicate_columns:
            raise CompilationError(f"Duplicate column_name values: {duplicate_columns}.")

    def _string_sequence(
        self,
        value: Any,
        label: str,
        *,
        allow_empty_values: bool = True,
    ) -> tuple[str, ...]:
        if not isinstance(value, list):
            raise CompilationError(f"{label} must be a YAML list of strings.")
        invalid = [
            item
            for item in value
            if not isinstance(item, str) or (not allow_empty_values and not item)
        ]
        if invalid:
            raise CompilationError(f"{label} must contain only valid strings: {invalid!r}.")
        return self._deduplicate(tuple(value))

    @staticmethod
    def _deduplicate(values: tuple[str, ...]) -> tuple[str, ...]:
        return tuple(dict.fromkeys(values))

    def _enum_value(self, enum_type: type, value: Any, label: str):
        if not isinstance(value, str):
            raise CompilationError(f"{label} must be a string.")
        try:
            return enum_type(value.lower())
        except ValueError as exc:
            valid = ", ".join(member.value for member in enum_type)
            raise CompilationError(f"Invalid {label} {value!r}. Valid values: {valid}.") from exc

    @staticmethod
    def _bool(payload: Mapping[str, Any], key: str, default: bool) -> bool:
        if key not in payload:
            return default
        value = payload[key]
        if not isinstance(value, bool):
            raise CompilationError(f"{key} must be true or false.")
        return value

    @staticmethod
    def _required_string(payload: Mapping[str, Any], key: str) -> str:
        value = payload.get(key)
        if not isinstance(value, str) or not value:
            raise CompilationError(f"{key} must be a non-empty string.")
        return value

    @staticmethod
    def _optional_string(payload: Mapping[str, Any], key: str) -> str | None:
        value = payload.get(key)
        if value is None:
            return None
        if not isinstance(value, str):
            raise CompilationError(f"{key} must be a string when provided.")
        return value

    @staticmethod
    def _ensure_mapping(value: Any, label: str) -> Mapping[str, Any]:
        if not isinstance(value, Mapping):
            raise CompilationError(f"{label} must be a mapping.")
        invalid_keys = [key for key in value if not isinstance(key, str)]
        if invalid_keys:
            raise CompilationError(f"{label} keys must be strings: {invalid_keys!r}.")
        return value

    @staticmethod
    def _reject_keys(payload: Mapping[str, Any], allowed: set[str], label: str) -> None:
        unsupported = sorted(set(payload) - allowed)
        if unsupported:
            raise CompilationError(f"{label} contains unsupported keys: {unsupported}.")
