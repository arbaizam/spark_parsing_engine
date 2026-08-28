"""Strict YAML compiler for parser configuration metadata."""

from __future__ import annotations

import base64
import binascii
import math
import re
from collections.abc import Mapping
from datetime import date, datetime
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any

import yaml

from spark_parser.data_types import SparkDataType, parse_spark_data_type
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
    DEFAULT_DATE_FORMATS,
    DEFAULT_DROP_NULL_ELEMENTS,
    DEFAULT_DROP_NULL_VALUES,
    DEFAULT_EMPTY_IS_NULL,
    DEFAULT_IS_NULLABLE,
    DEFAULT_NULL_MARKER_CASE_SENSITIVE,
    DEFAULT_NULL_MARKERS,
    DEFAULT_NULL_MARKERS_MODE,
    DEFAULT_ON_PARSE_ERROR,
    DEFAULT_REPLACE_NULL_MARKERS,
    DEFAULT_TIMESTAMP_FORMATS,
    DEFAULT_TIMESTAMP_NTZ_FORMATS,
    DEFAULT_TRIM_WHITESPACE,
    DEFAULT_ZERO_IS_VALID,
)
from spark_parser.enums import (
    COMPLEX_PARSER_TYPES,
    NUMERIC_PARSER_TYPES,
    BinaryEncoding,
    BooleanValuesMode,
    ChildErrorMode,
    ComplexInputFormat,
    NullMarkersMode,
    ParseErrorMode,
    ParserType,
    StringFormat,
)
from spark_parser.exceptions import CompilationError
from spark_parser.models import (
    ColumnParser,
    NestedValueParser,
    ParserConfig,
    ParserGlobals,
    ParserOptions,
    StructFieldParser,
)

_MISSING = object()
_DECIMAL_PATTERN = re.compile(r"decimal\s*\(\s*(\d+)\s*,\s*(\d+)\s*\)", re.IGNORECASE)
_TYPE_ALIASES = {
    "tinyint": "byte",
    "smallint": "short",
    "int": "integer",
    "bigint": "long",
    "real": "float",
    "bool": "boolean",
    "timestamp_ltz": "timestamp",
}
_BYTE_MIN = -(2**7)
_BYTE_MAX = 2**7 - 1
_SHORT_MIN = -(2**15)
_SHORT_MAX = 2**15 - 1
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
        payload = self._ensure_mapping(payload, "parser config")
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
            {
                "null_markers",
                "null_marker_case_sensitive",
                "true_values",
                "false_values",
                "boolean_case_sensitive",
            },
            "globals",
        )
        true_values = self._string_sequence(
            payload.get("true_values", list(DEFAULT_BOOLEAN_TRUE_VALUES)),
            "globals.true_values",
            allow_empty_values=False,
        )
        false_values = self._string_sequence(
            payload.get("false_values", list(DEFAULT_BOOLEAN_FALSE_VALUES)),
            "globals.false_values",
            allow_empty_values=False,
        )
        if not true_values or not false_values:
            raise CompilationError(
                "globals.true_values and globals.false_values must be non-empty."
            )
        boolean_case_sensitive = self._bool(
            payload,
            "boolean_case_sensitive",
            DEFAULT_BOOLEAN_CASE_SENSITIVE,
        )
        self._validate_boolean_overlap(
            true_values,
            false_values,
            boolean_case_sensitive,
            "globals",
        )
        return ParserGlobals(
            null_markers=self._string_sequence(
                payload.get("null_markers", list(DEFAULT_NULL_MARKERS)),
                "null_markers",
            ),
            null_marker_case_sensitive=self._bool(
                payload,
                "null_marker_case_sensitive",
                DEFAULT_NULL_MARKER_CASE_SENSITIVE,
            ),
            true_values=true_values,
            false_values=false_values,
            boolean_case_sensitive=boolean_case_sensitive,
        )

    def _compile_column(
        self,
        raw_column: Any,
        globals_config: ParserGlobals,
        index: int,
    ) -> ColumnParser:
        payload = self._ensure_mapping(raw_column, f"column at index {index}")
        legacy_keys = sorted({"column_name", "data_type"} & set(payload))
        if legacy_keys:
            raise CompilationError(
                f"Column at index {index} uses 0.2.x keys {legacy_keys}. In 0.3.0, "
                "replace column_name with both source_column_name and silver_column_name, "
                "and replace data_type with expected_data_type."
            )
        self._reject_keys(
            payload,
            {
                "source_column_name",
                "silver_column_name",
                "expected_data_type",
                "parser",
            },
            f"Column at index {index}",
        )
        source_column_name = self._required_string(payload, "source_column_name")
        silver_column_name = self._required_string(payload, "silver_column_name")
        data_type = parse_spark_data_type(self._required_string(payload, "expected_data_type"))
        expected_data_type = data_type.canonical
        options = self._compile_parser(
            payload.get("parser", _MISSING),
            globals_config,
            data_type,
            silver_column_name,
            allow_audit=True,
        )
        return ColumnParser(
            source_column_name=source_column_name,
            silver_column_name=silver_column_name,
            expected_data_type=expected_data_type,
            data_type=data_type,
            parser=options,
        )

    def _compile_parser(
        self,
        raw_parser: Any,
        globals_config: ParserGlobals,
        data_type: SparkDataType,
        silver_column_name: str,
        *,
        allow_audit: bool,
        child_error_owned_by_parent: bool = False,
    ) -> ParserOptions:
        if raw_parser is _MISSING:
            raise CompilationError(f"parser is required for column {silver_column_name!r}.")
        if isinstance(raw_parser, str):
            payload: Mapping[str, Any] = {"type": raw_parser}
        else:
            payload = self._ensure_mapping(raw_parser, f"parser for {silver_column_name!r}")

        parser_type = self._parser_type(self._required_string(payload, "type"))
        expected_data_type = data_type.canonical
        if parser_type is not data_type.parser_type:
            raise CompilationError(
                f"Parser {parser_type.value!r} is incompatible with expected_data_type "
                f"{expected_data_type!r} for silver column {silver_column_name!r}; "
                f"expected {data_type.parser_type.value!r}."
            )
        self._validate_nested_parser_contract(
            payload,
            silver_column_name,
            allow_audit=allow_audit,
            child_error_owned_by_parent=child_error_owned_by_parent,
        )
        allowed_keys = self._parser_allowed_keys(parser_type, allow_audit=allow_audit)
        self._reject_keys(payload, allowed_keys, f"Parser for {silver_column_name!r}")

        markers_mode = self._enum_value(
            NullMarkersMode,
            payload.get("null_markers_mode", DEFAULT_NULL_MARKERS_MODE.value),
            "null_markers_mode",
        )
        if "null_markers_mode" in payload and "null_markers" not in payload:
            raise CompilationError(
                f"null_markers_mode for {silver_column_name!r} requires column null_markers."
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
        replace_null_markers = self._bool(
            payload,
            "replace_null_markers",
            DEFAULT_REPLACE_NULL_MARKERS,
        )
        if replace_null_markers and not markers:
            raise CompilationError(
                f"replace_null_markers is true for {silver_column_name!r}, "
                "but no null markers exist."
            )

        is_nullable = self._bool(payload, "is_nullable", DEFAULT_IS_NULLABLE)
        raw_default_on_null = payload.get("default_on_null", _MISSING)
        if not is_nullable and raw_default_on_null is _MISSING:
            raise CompilationError(
                f"Column {silver_column_name!r} is not nullable and requires default_on_null."
            )
        if is_nullable and raw_default_on_null is not _MISSING:
            raise CompilationError(
                f"default_on_null for {silver_column_name!r} requires is_nullable: false."
            )
        default_on_null = (
            self._typed_default(
                raw_default_on_null,
                data_type,
                "default_on_null",
            )
            if raw_default_on_null is not _MISSING
            else None
        )

        raw_on_parse_error = payload.get("on_parse_error", DEFAULT_ON_PARSE_ERROR.value)
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
                f"on_parse_error: default for {silver_column_name!r} requires default_on_error."
            )
        if on_parse_error is not ParseErrorMode.DEFAULT and raw_default_on_error is not _MISSING:
            raise CompilationError(
                f"default_on_error for {silver_column_name!r} requires on_parse_error: default."
            )
        default_on_error = (
            self._typed_default(
                raw_default_on_error,
                data_type,
                "default_on_error",
            )
            if raw_default_on_error is not _MISSING
            else None
        )

        zero_is_valid = self._bool(payload, "zero_is_valid", DEFAULT_ZERO_IS_VALID)
        if (
            parser_type in NUMERIC_PARSER_TYPES
            and not zero_is_valid
            and default_on_null is not None
            and Decimal(str(default_on_null)) == 0
        ):
            raise CompilationError(
                f"Column {silver_column_name!r} rejects zero but uses zero as default_on_null."
            )
        if (
            parser_type in NUMERIC_PARSER_TYPES
            and not zero_is_valid
            and default_on_error is not None
            and Decimal(str(default_on_error)) == 0
        ):
            raise CompilationError(
                f"Column {silver_column_name!r} rejects zero but uses zero as default_on_error."
            )

        string_format = self._compile_string_format(payload, parser_type)
        formats = self._compile_formats(payload, parser_type)
        (
            true_values,
            false_values,
            boolean_case_sensitive,
            boolean_values_mode,
        ) = self._compile_boolean_values(
            payload,
            parser_type,
            silver_column_name,
            globals_config,
        )
        complex_options = self._compile_complex_options(
            payload,
            data_type,
            globals_config,
            silver_column_name,
        )
        compiled_options = ParserOptions(
            parser_type=parser_type,
            trim_whitespace=self._bool(
                payload,
                "trim_whitespace",
                DEFAULT_TRIM_WHITESPACE,
            ),
            collapse_whitespace=self._bool(
                payload,
                "collapse_whitespace",
                DEFAULT_COLLAPSE_WHITESPACE,
            ),
            empty_is_null=self._bool(payload, "empty_is_null", DEFAULT_EMPTY_IS_NULL),
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
            audit=self._bool(payload, "audit", DEFAULT_AUDIT) if allow_audit else False,
            zero_is_valid=zero_is_valid,
            string_format=string_format,
            formats=formats,
            true_values=true_values,
            false_values=false_values,
            boolean_case_sensitive=boolean_case_sensitive,
            boolean_values_mode=boolean_values_mode,
            binary_encoding=self._enum_value(
                BinaryEncoding,
                payload.get("encoding", DEFAULT_BINARY_ENCODING.value),
                "encoding",
            ),
            **complex_options,
        )
        self._validate_binary_defaults(compiled_options, data_type)
        return compiled_options

    def _validate_binary_defaults(
        self,
        options: ParserOptions,
        data_type: SparkDataType,
    ) -> None:
        if options.default_on_null is not None:
            self._validate_binary_value(
                options.default_on_null,
                data_type,
                options,
                "default_on_null",
            )
        if options.default_on_error is not None:
            self._validate_binary_value(
                options.default_on_error,
                data_type,
                options,
                "default_on_error",
            )

    def _validate_binary_value(
        self,
        value: Any,
        data_type: SparkDataType,
        options: ParserOptions,
        label: str,
    ) -> None:
        if value is None:
            return
        if data_type.parser_type is ParserType.BINARY:
            try:
                if options.binary_encoding is BinaryEncoding.BASE64:
                    base64.b64decode(value, validate=True)
                elif options.binary_encoding is BinaryEncoding.HEX:
                    bytes.fromhex(value)
                else:
                    value.encode("utf-8")
            except (ValueError, UnicodeError, binascii.Error) as exc:
                raise CompilationError(
                    f"{label} is not valid {options.binary_encoding.value} binary text."
                ) from exc
            return
        if data_type.parser_type is ParserType.ARRAY:
            assert data_type.element_type is not None and options.element_parser is not None
            for index, item in enumerate(value):
                self._validate_binary_value(
                    item,
                    data_type.element_type,
                    options.element_parser.parser,
                    f"{label}[{index}]",
                )
            return
        if data_type.parser_type is ParserType.STRUCT:
            for field in options.field_parsers:
                self._validate_binary_value(
                    value[field.silver_field_name],
                    field.data_type,
                    field.parser,
                    f"{label}.{field.silver_field_name}",
                )
            return
        if data_type.parser_type is ParserType.MAP:
            assert data_type.value_type is not None and options.value_parser is not None
            for key, item in value.items():
                self._validate_binary_value(
                    item,
                    data_type.value_type,
                    options.value_parser.parser,
                    f"{label}[{key!r}]",
                )

    @staticmethod
    def _validate_nested_parser_contract(
        payload: Mapping[str, Any],
        label: str,
        *,
        allow_audit: bool,
        child_error_owned_by_parent: bool,
    ) -> None:
        if not allow_audit and "audit" in payload:
            raise CompilationError(
                f"Nested parser {label!r} cannot enable audit; audit belongs to its "
                "configured top-level column."
            )
        if child_error_owned_by_parent and (
            "on_parse_error" in payload or "default_on_error" in payload
        ):
            raise CompilationError(
                f"Nested parser {label!r} is controlled by its parent child-error policy "
                "and cannot set on_parse_error or default_on_error."
            )

    @staticmethod
    def _parser_allowed_keys(parser_type: ParserType, *, allow_audit: bool) -> set[str]:
        common = {
            "type",
            "trim_whitespace",
            "collapse_whitespace",
            "empty_is_null",
            "replace_null_markers",
            "null_markers",
            "null_markers_mode",
            "null_marker_case_sensitive",
            "is_nullable",
            "default_on_null",
            "on_parse_error",
            "default_on_error",
        }
        if allow_audit:
            common.add("audit")
        specific = {
            ParserType.STRING: {"format"},
            ParserType.BYTE: {"zero_is_valid"},
            ParserType.SHORT: {"zero_is_valid"},
            ParserType.INTEGER: {"zero_is_valid"},
            ParserType.LONG: {"zero_is_valid"},
            ParserType.FLOAT: {"zero_is_valid"},
            ParserType.DECIMAL: {"zero_is_valid"},
            ParserType.DOUBLE: {"zero_is_valid"},
            ParserType.BINARY: {"encoding"},
            ParserType.BOOLEAN: {
                "true_values",
                "false_values",
                "boolean_case_sensitive",
                "boolean_values_mode",
            },
            ParserType.DATE: {"formats"},
            ParserType.TIMESTAMP: {"formats"},
            ParserType.TIMESTAMP_NTZ: {"formats"},
            ParserType.ARRAY: {
                "input_format",
                "delimiter",
                "element_parser",
                "on_element_error",
                "drop_null_elements",
                "distinct",
            },
            ParserType.STRUCT: {"input_format", "fields"},
            ParserType.MAP: {
                "input_format",
                "value_parser",
                "on_value_error",
                "drop_null_values",
            },
        }
        return common | specific[parser_type]

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
            default = DEFAULT_DATE_FORMATS
        elif parser_type is ParserType.TIMESTAMP:
            default = DEFAULT_TIMESTAMP_FORMATS
        elif parser_type is ParserType.TIMESTAMP_NTZ:
            default = DEFAULT_TIMESTAMP_NTZ_FORMATS
        else:
            return ()
        if "formats" not in payload:
            return default
        formats = self._string_sequence(payload["formats"], "formats", allow_empty_values=False)
        if not formats:
            raise CompilationError("formats must contain at least one Spark datetime pattern.")
        return formats

    def _compile_complex_options(
        self,
        payload: Mapping[str, Any],
        data_type: SparkDataType,
        globals_config: ParserGlobals,
        label: str,
    ) -> dict[str, Any]:
        defaults: dict[str, Any] = {
            "input_format": DEFAULT_COMPLEX_INPUT_FORMAT,
            "delimiter": None,
            "element_parser": None,
            "field_parsers": (),
            "value_parser": None,
            "on_element_error": DEFAULT_CHILD_ERROR_MODE,
            "on_value_error": DEFAULT_CHILD_ERROR_MODE,
            "drop_null_elements": DEFAULT_DROP_NULL_ELEMENTS,
            "distinct": DEFAULT_ARRAY_DISTINCT,
            "drop_null_values": DEFAULT_DROP_NULL_VALUES,
        }
        parser_type = data_type.parser_type
        if parser_type not in COMPLEX_PARSER_TYPES:
            return defaults

        input_format = self._enum_value(
            ComplexInputFormat,
            payload.get("input_format", DEFAULT_COMPLEX_INPUT_FORMAT.value),
            "input_format",
        )
        if parser_type in {ParserType.STRUCT, ParserType.MAP} and (
            input_format is not ComplexInputFormat.JSON
        ):
            raise CompilationError(
                f"{parser_type.value} parser {label!r} supports only JSON input."
            )
        defaults["input_format"] = input_format

        if parser_type is ParserType.ARRAY:
            assert data_type.element_type is not None
            raw_element_parser = payload.get("element_parser", _MISSING)
            if raw_element_parser is _MISSING:
                raise CompilationError(f"Array parser {label!r} requires element_parser.")
            if input_format is ComplexInputFormat.DELIMITED:
                if data_type.element_type.is_complex:
                    raise CompilationError(
                        f"Delimited array parser {label!r} requires a scalar element type."
                    )
                delimiter = payload.get("delimiter")
                if not isinstance(delimiter, str) or not delimiter:
                    raise CompilationError(
                        f"delimiter for array parser {label!r} must be a non-empty string."
                    )
                defaults["delimiter"] = delimiter
            elif "delimiter" in payload:
                raise CompilationError(
                    f"delimiter for array parser {label!r} requires input_format: delimited."
                )
            element_options = self._compile_parser(
                raw_element_parser,
                globals_config,
                data_type.element_type,
                f"{label}[]",
                allow_audit=False,
                child_error_owned_by_parent=True,
            )
            distinct = self._bool(payload, "distinct", DEFAULT_ARRAY_DISTINCT)
            if distinct and not data_type.element_type.supports_equality:
                raise CompilationError(
                    f"Array parser {label!r} cannot use distinct with non-comparable element "
                    f"type {data_type.element_type.canonical!r}."
                )
            defaults.update(
                element_parser=NestedValueParser(
                    expected_data_type=data_type.element_type.canonical,
                    data_type=data_type.element_type,
                    parser=element_options,
                ),
                on_element_error=self._child_error_mode(payload, "on_element_error"),
                drop_null_elements=self._bool(
                    payload,
                    "drop_null_elements",
                    DEFAULT_DROP_NULL_ELEMENTS,
                ),
                distinct=distinct,
            )
            return defaults

        if parser_type is ParserType.STRUCT:
            raw_fields = payload.get("fields")
            if not isinstance(raw_fields, list) or not raw_fields:
                raise CompilationError(f"Struct parser {label!r} requires a non-empty fields list.")
            expected_fields = {field.name: field.data_type for field in data_type.fields}
            compiled_by_name: dict[str, StructFieldParser] = {}
            source_names: list[str] = []
            for index, raw_field in enumerate(raw_fields, start=1):
                field_payload = self._ensure_mapping(
                    raw_field,
                    f"field {index} for struct parser {label!r}",
                )
                self._reject_keys(
                    field_payload,
                    {"source_field_name", "silver_field_name", "parser"},
                    f"Field {index} for struct parser {label!r}",
                )
                source_name = self._required_string(field_payload, "source_field_name")
                silver_name = self._required_string(field_payload, "silver_field_name")
                if silver_name not in expected_fields:
                    raise CompilationError(
                        f"Struct parser {label!r} configures unknown silver field "
                        f"{silver_name!r}; expected {sorted(expected_fields)}."
                    )
                if silver_name in compiled_by_name:
                    raise CompilationError(
                        f"Struct parser {label!r} has duplicate silver field {silver_name!r}."
                    )
                field_type = expected_fields[silver_name]
                field_options = self._compile_parser(
                    field_payload.get("parser", _MISSING),
                    globals_config,
                    field_type,
                    f"{label}.{silver_name}",
                    allow_audit=False,
                )
                compiled_by_name[silver_name] = StructFieldParser(
                    source_field_name=source_name,
                    silver_field_name=silver_name,
                    expected_data_type=field_type.canonical,
                    data_type=field_type,
                    parser=field_options,
                )
                source_names.append(source_name)
            missing_fields = sorted(set(expected_fields) - set(compiled_by_name))
            if missing_fields:
                raise CompilationError(
                    f"Struct parser {label!r} is missing field configs for {missing_fields}."
                )
            duplicate_sources = sorted(
                {name for name in source_names if source_names.count(name) > 1}
            )
            if duplicate_sources:
                raise CompilationError(
                    f"Struct parser {label!r} has duplicate source fields {duplicate_sources}."
                )
            defaults["field_parsers"] = tuple(
                compiled_by_name[field.name] for field in data_type.fields
            )
            return defaults

        assert parser_type is ParserType.MAP
        assert data_type.key_type is not None and data_type.value_type is not None
        if data_type.key_type.parser_type is not ParserType.STRING:
            raise CompilationError(
                f"JSON map parser {label!r} requires map<string,...>; found "
                f"{data_type.key_type.canonical} keys."
            )
        raw_value_parser = payload.get("value_parser", _MISSING)
        if raw_value_parser is _MISSING:
            raise CompilationError(f"Map parser {label!r} requires value_parser.")
        value_options = self._compile_parser(
            raw_value_parser,
            globals_config,
            data_type.value_type,
            f"{label}{{value}}",
            allow_audit=False,
            child_error_owned_by_parent=True,
        )
        defaults.update(
            value_parser=NestedValueParser(
                expected_data_type=data_type.value_type.canonical,
                data_type=data_type.value_type,
                parser=value_options,
            ),
            on_value_error=self._child_error_mode(payload, "on_value_error"),
            drop_null_values=self._bool(
                payload,
                "drop_null_values",
                DEFAULT_DROP_NULL_VALUES,
            ),
        )
        return defaults

    def _child_error_mode(
        self,
        payload: Mapping[str, Any],
        key: str,
    ) -> ChildErrorMode:
        raw_value = payload.get(key, DEFAULT_CHILD_ERROR_MODE.value)
        if key in payload and raw_value is None:
            raw_value = ChildErrorMode.NULL.value
        return self._enum_value(ChildErrorMode, raw_value, key)

    def _compile_boolean_values(
        self,
        payload: Mapping[str, Any],
        parser_type: ParserType,
        silver_column_name: str,
        globals_config: ParserGlobals,
    ) -> tuple[tuple[str, ...], tuple[str, ...], bool, BooleanValuesMode]:
        if parser_type is not ParserType.BOOLEAN:
            return (
                DEFAULT_BOOLEAN_TRUE_VALUES,
                DEFAULT_BOOLEAN_FALSE_VALUES,
                DEFAULT_BOOLEAN_CASE_SENSITIVE,
                DEFAULT_BOOLEAN_VALUES_MODE,
            )
        mode = self._enum_value(
            BooleanValuesMode,
            payload.get("boolean_values_mode", DEFAULT_BOOLEAN_VALUES_MODE.value),
            "boolean_values_mode",
        )
        supplied_true = "true_values" in payload
        supplied_false = "false_values" in payload
        if "boolean_values_mode" in payload and not (supplied_true or supplied_false):
            raise CompilationError(
                f"boolean_values_mode for {silver_column_name!r} requires true_values "
                "or false_values."
            )
        column_true = (
            self._string_sequence(
                payload["true_values"],
                "true_values",
                allow_empty_values=False,
            )
            if supplied_true
            else ()
        )
        column_false = (
            self._string_sequence(
                payload["false_values"],
                "false_values",
                allow_empty_values=False,
            )
            if supplied_false
            else ()
        )
        if mode is BooleanValuesMode.EXTEND:
            true_values = self._deduplicate((*globals_config.true_values, *column_true))
            false_values = self._deduplicate((*globals_config.false_values, *column_false))
        else:
            true_values = column_true if supplied_true else globals_config.true_values
            false_values = column_false if supplied_false else globals_config.false_values
        if not true_values or not false_values:
            raise CompilationError("true_values and false_values must be non-empty.")
        case_sensitive = self._bool(
            payload,
            "boolean_case_sensitive",
            globals_config.boolean_case_sensitive,
        )
        self._validate_boolean_overlap(
            true_values,
            false_values,
            case_sensitive,
            f"silver column {silver_column_name!r}",
        )
        return true_values, false_values, case_sensitive, mode

    @staticmethod
    def _validate_boolean_overlap(
        true_values: tuple[str, ...],
        false_values: tuple[str, ...],
        case_sensitive: bool,
        label: str,
    ) -> None:
        normalize = (lambda item: item) if case_sensitive else (lambda item: item.lower())
        overlap = {normalize(item) for item in true_values} & {
            normalize(item) for item in false_values
        }
        if overlap:
            raise CompilationError(
                f"Boolean true_values and false_values overlap for {label}: {sorted(overlap)}."
            )

    def _normalize_data_type(self, value: str) -> tuple[str, ParserType]:
        data_type = parse_spark_data_type(value)
        return data_type.canonical, data_type.parser_type

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
        data_type: SparkDataType,
        label: str,
    ) -> Any:
        if value is None:
            raise CompilationError(f"{label} must be non-null.")
        parser_type = data_type.parser_type
        if parser_type is ParserType.STRING:
            if not isinstance(value, str):
                raise CompilationError(f"{label} for string must be a string.")
            return value
        if parser_type in {
            ParserType.BYTE,
            ParserType.SHORT,
            ParserType.INTEGER,
            ParserType.LONG,
        }:
            return self._integer_default(value, parser_type, label)
        if parser_type in {ParserType.FLOAT, ParserType.DOUBLE}:
            return self._double_default(value, label)
        if parser_type is ParserType.DECIMAL:
            return self._decimal_default(value, data_type.canonical, label)
        if parser_type is ParserType.BOOLEAN:
            if not isinstance(value, bool):
                raise CompilationError(f"{label} for boolean must be true or false.")
            return value
        if parser_type is ParserType.DATE:
            return self._date_default(value, label)
        if parser_type in {ParserType.TIMESTAMP, ParserType.TIMESTAMP_NTZ}:
            return self._timestamp_default(value, label)
        if parser_type is ParserType.BINARY:
            if not isinstance(value, str):
                raise CompilationError(f"{label} for binary must be an encoded string.")
            return value
        if parser_type is ParserType.ARRAY:
            if not isinstance(value, list):
                raise CompilationError(f"{label} for array must be a YAML list.")
            assert data_type.element_type is not None
            return [
                None if item is None else self._typed_default(item, data_type.element_type, label)
                for item in value
            ]
        if parser_type is ParserType.STRUCT:
            mapping = self._ensure_mapping(value, f"{label} for struct")
            expected = {field.name: field.data_type for field in data_type.fields}
            unknown = sorted(set(mapping) - set(expected))
            missing = sorted(set(expected) - set(mapping))
            if unknown or missing:
                raise CompilationError(
                    f"{label} for struct has missing fields {missing} and unknown fields {unknown}."
                )
            return {
                name: (
                    None
                    if mapping[name] is None
                    else self._typed_default(mapping[name], field_type, f"{label}.{name}")
                )
                for name, field_type in expected.items()
            }
        if parser_type is ParserType.MAP:
            mapping = self._ensure_mapping(value, f"{label} for map")
            assert data_type.key_type is not None and data_type.value_type is not None
            if data_type.key_type.parser_type is not ParserType.STRING:
                raise CompilationError(f"{label} supports only string-keyed map defaults.")
            return {
                key: (
                    None
                    if item is None
                    else self._typed_default(item, data_type.value_type, f"{label}[{key!r}]")
                )
                for key, item in mapping.items()
            }
        raise CompilationError(f"Unsupported parser type for {label}: {parser_type.value}.")

    @staticmethod
    def _integer_default(value: Any, parser_type: ParserType, label: str) -> int:
        if isinstance(value, bool) or not isinstance(value, int):
            raise CompilationError(f"{label} for {parser_type.value} must be an integer.")
        ranges = {
            ParserType.BYTE: (_BYTE_MIN, _BYTE_MAX),
            ParserType.SHORT: (_SHORT_MIN, _SHORT_MAX),
            ParserType.INTEGER: (_INTEGER_MIN, _INTEGER_MAX),
            ParserType.LONG: (_LONG_MIN, _LONG_MAX),
        }
        minimum, maximum = ranges[parser_type]
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
        column_names = [column.silver_column_name for column in columns]
        duplicate_columns = sorted({name for name in column_names if column_names.count(name) > 1})
        if duplicate_columns:
            raise CompilationError(f"Duplicate silver_column_name values: {duplicate_columns}.")

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
        if not isinstance(value, str) or not value.strip():
            raise CompilationError(f"{key} must be a non-empty string.")
        return value.strip()

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
