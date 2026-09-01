"""Compile scalar YAML authoring metadata into an executable parser contract.

This module is the package's trust boundary. It rejects ambiguity early, resolves every inherited
or omitted option, validates typed defaults, and returns immutable models that runtime code can use
without defensive revalidation. Compilation never requires a Spark session.
"""

from __future__ import annotations

import base64
import binascii
import math
import re
import struct
from collections.abc import Mapping
from datetime import date, datetime, timezone
from decimal import Decimal, InvalidOperation
from enum import Enum
from pathlib import Path
from typing import Any, TypeVar

import yaml

from spark_parser.data_types import SparkDataType, canonical_type_name, parse_spark_data_type
from spark_parser.defaults import (
    DEFAULT_AUDIT,
    DEFAULT_BINARY_ENCODING,
    DEFAULT_BOOLEAN_CASE_SENSITIVE,
    DEFAULT_BOOLEAN_FALSE_VALUES,
    DEFAULT_BOOLEAN_TRUE_VALUES,
    DEFAULT_BOOLEAN_VALUES_MODE,
    DEFAULT_COLLAPSE_WHITESPACE,
    DEFAULT_DATE_FORMATS,
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
    NUMERIC_PARSER_TYPES,
    BinaryEncoding,
    BooleanValuesMode,
    NullMarkersMode,
    ParseErrorMode,
    ParserType,
    StringFormat,
)
from spark_parser.exceptions import CompilationError
from spark_parser.models import ColumnParser, ParserConfig, ParserGlobals, ParserOptions

# ``None`` is a legitimate YAML value, so a private sentinel is required to distinguish an omitted
# key from an explicitly authored null.
_MISSING = object()

# Spark integer types have fixed signed ranges. Python integers do not overflow, which means these
# boundaries must be enforced here before defaults become Spark literals.
_BYTE_MIN = -(2**7)
_BYTE_MAX = 2**7 - 1
_SHORT_MIN = -(2**15)
_SHORT_MAX = 2**15 - 1
_INTEGER_MIN = -(2**31)
_INTEGER_MAX = 2**31 - 1
_LONG_MIN = -(2**63)
_LONG_MAX = 2**63 - 1

# Keep validation independent of permissive or version-dependent standard-library ISO parsing.
_DATE_DEFAULT_PATTERN = re.compile(r"[0-9]{4}-[0-9]{2}-[0-9]{2}")
_TIMESTAMP_DEFAULT_PATTERN = re.compile(
    r"[0-9]{4}-[0-9]{2}-[0-9]{2}T"
    r"[0-9]{2}:[0-9]{2}:[0-9]{2}"
    r"(?:\.[0-9]{1,6})?"
    r"(?:Z|[+-](?:(?:0[0-9]|1[0-7]):[0-5][0-9]|18:00))?"
)
_DECIMAL_DEFAULT_PATTERN = re.compile(r"[+-]?(?:[0-9]+(?:\.[0-9]*)?|\.[0-9]+)(?:[eE][+-]?[0-9]+)?")
_ASCII_HEX_PATTERN = re.compile(r"[0-9A-Fa-f]*")
# Composer recursion happens before constructor validation. Bound it separately so deeply nested
# untrusted YAML cannot depend on the host interpreter's recursion limit or leak RecursionError.
_MAX_YAML_COMPOSE_DEPTH = 256
_EnumT = TypeVar("_EnumT", bound=Enum)


def _validate_utf8_string(value: str, label: str) -> str:
    """Require a Python string that can be represented as well-formed UTF-8."""
    try:
        value.encode("utf-8")
    except UnicodeEncodeError as exc:
        raise CompilationError(
            f"{label} must contain well-formed Unicode; "
            f"invalid code point at character {exc.start + 1}."
        ) from exc
    return value


def _has_unquoted_timezone_pattern(pattern: str) -> bool:
    """Detect Spark zone/offset fields while respecting quoted pattern literals."""
    in_literal = False
    index = 0
    while index < len(pattern):
        character = pattern[index]
        if character == "'":
            if index + 1 < len(pattern) and pattern[index + 1] == "'":
                index += 2
                continue
            in_literal = not in_literal
        elif not in_literal and character in "VvzOXxZ":
            return True
        index += 1
    return False


class _UniqueKeyLoader(yaml.SafeLoader):
    """Safe YAML loader that rejects duplicate mapping keys.

    Standard YAML loaders silently keep one duplicate value. That is dangerous for parsing rules:
    a reviewer may approve one value while the loader executes another.
    """

    def __init__(self, stream: Any) -> None:
        """Initialize composition-depth tracking before PyYAML starts reading the stream."""
        self._compose_depth = 0
        super().__init__(stream)

    def compose_node(self, parent: yaml.Node | None, index: int) -> yaml.Node:
        """Compose one YAML node while enforcing a deterministic nesting bound."""
        if self._compose_depth >= _MAX_YAML_COMPOSE_DEPTH:
            mark = self.peek_event().start_mark
            raise CompilationError(
                "YAML nesting exceeds the maximum depth of "
                f"{_MAX_YAML_COMPOSE_DEPTH} at line {mark.line + 1}, "
                f"column {mark.column + 1}."
            )
        self._compose_depth += 1
        try:
            composed = super().compose_node(parent, index)
            assert composed is not None
            return composed
        finally:
            self._compose_depth -= 1


def _reject_yaml_merge(
    _loader: _UniqueKeyLoader,
    _node: yaml.Node,
) -> None:
    """Reject YAML's ``<<`` merge operator with an actionable policy explanation.

    PyYAML's ordinary error mentions an internal constructor tag. That fails closed, but it does not
    tell a configuration author that merge keys are intentionally excluded from reviewable configs.
    """
    raise CompilationError(
        "YAML merge keys (<<) are not supported because they hide inherited keys from review; "
        "use a plain anchor/alias or repeat the keys explicitly."
    )


def _construct_timestamp_string(
    loader: _UniqueKeyLoader,
    node: yaml.ScalarNode,
) -> str:
    """Preserve YAML timestamps as text for the compiler's context-specific strict grammar."""
    return loader.construct_scalar(node)


def _construct_unique_mapping(
    loader: _UniqueKeyLoader,
    node: yaml.MappingNode,
    deep: bool = False,
) -> dict[Any, Any]:
    """Construct one YAML mapping while detecting duplicate keys at its current depth."""
    mapping: dict[Any, Any] = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        location = f"line {key_node.start_mark.line + 1}, column {key_node.start_mark.column + 1}"
        if isinstance(key, str):
            _validate_utf8_string(key, f"YAML mapping key at {location}")
        try:
            hash(key)
        except TypeError as exc:
            raise CompilationError(f"YAML mapping key must be hashable at {location}.") from exc
        if key in mapping:
            raise CompilationError(f"Duplicate YAML key at {location}: {key!r}.")
        mapping[key] = loader.construct_object(value_node, deep=deep)
    return mapping


# Register the strict constructor for every ordinary YAML mapping, including nested parser maps.
_UniqueKeyLoader.add_constructor(
    yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG,
    _construct_unique_mapping,
)
_UniqueKeyLoader.add_constructor("tag:yaml.org,2002:merge", _reject_yaml_merge)
_UniqueKeyLoader.add_constructor("tag:yaml.org,2002:timestamp", _construct_timestamp_string)


class YamlParserConfigCompiler:
    """Compile strict authoring input into a fully resolved :class:`ParserConfig`.

    Public methods accept a file, text, or already-loaded mapping. All routes converge on
    :meth:`compile_mapping`, so validation behavior cannot diverge by input format.
    """

    def compile_path(self, path: str | Path) -> ParserConfig:
        """Read and compile one UTF-8 YAML file, preserving the operating-system error cause."""
        try:
            text = Path(path).read_text(encoding="utf-8")
        except UnicodeError as exc:
            raise CompilationError(
                f"Unable to decode parser config {path!s} as well-formed UTF-8: {exc}"
            ) from exc
        except (OSError, ValueError) as exc:
            raise CompilationError(f"Unable to read parser config {path!s}: {exc}") from exc
        return self.compile_text(text)

    def compile_text(self, text: str) -> ParserConfig:
        """Load one YAML document with duplicate-key protection, then compile its mapping."""
        if not isinstance(text, str):
            raise CompilationError("parser YAML must be text.")
        _validate_utf8_string(text, "parser YAML")
        try:
            payload = yaml.load(text, Loader=_UniqueKeyLoader)
        except CompilationError:
            raise
        except yaml.YAMLError as exc:
            raise CompilationError(f"Invalid parser YAML: {exc}") from exc
        except (RecursionError, ValueError) as exc:
            raise CompilationError(f"Invalid parser YAML: {exc}") from exc
        return self.compile_mapping(self._ensure_mapping(payload, "parser config"))

    def compile_mapping(self, payload: Mapping[str, Any]) -> ParserConfig:
        """Validate and fully resolve an already-loaded YAML-compatible mapping.

        Validation proceeds from the outside inward: top-level keys, inherited globals, each
        column/parser binding, and finally cross-column uniqueness. The returned object therefore
        represents a complete runtime contract rather than partially validated authoring data.
        """
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
        # Globals must be compiled before columns because column vocabularies can inherit or extend
        # them. No column receives the mutable raw mapping.
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
        """Validate global null/Boolean vocabularies and return immutable inherited options."""
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
        # Overlap would make the same bronze token both true and false. Validate using exactly the
        # case-folding behavior that Spark expressions use later.
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
        """Compile one top-level source-to-target scalar mapping."""
        payload = self._ensure_mapping(raw_column, f"column at index {index}")
        self._reject_keys(
            payload,
            {
                "source_column_name",
                "target_column_name",
                "expected_data_type",
                "parser",
            },
            f"Column at index {index}",
        )
        source_column_name = self._required_identifier(payload, "source_column_name")
        target_column_name = self._required_identifier(payload, "target_column_name")
        # Parse the scalar DDL once and carry both its model and canonical text. Complex DDL is
        # rejected by ``parse_spark_data_type`` before parser options are considered.
        data_type = parse_spark_data_type(self._required_string(payload, "expected_data_type"))
        expected_data_type = data_type.canonical
        options = self._compile_parser(
            payload.get("parser", _MISSING),
            globals_config,
            data_type,
            target_column_name,
        )
        return ColumnParser(
            source_column_name=source_column_name,
            target_column_name=target_column_name,
            expected_data_type=expected_data_type,
            data_type=data_type,
            parser=options,
        )

    def _compile_parser(
        self,
        raw_parser: Any,
        globals_config: ParserGlobals,
        data_type: SparkDataType,
        target_column_name: str,
    ) -> ParserOptions:
        """Resolve one scalar parser against its expected datatype."""
        if raw_parser is _MISSING:
            raise CompilationError(f"parser is required for column {target_column_name!r}.")
        # Scalar shorthand such as ``parser: date`` is normalized into the same mapping path as
        # long-form options. From this point onward there is only one validation implementation.
        if isinstance(raw_parser, str):
            payload: Mapping[str, Any] = {"type": raw_parser}
        else:
            payload = self._ensure_mapping(raw_parser, f"parser for {target_column_name!r}")

        parser_type = self._parser_type(self._required_string(payload, "type"))
        expected_data_type = data_type.canonical
        if parser_type is not data_type.parser_type:
            raise CompilationError(
                f"Parser {parser_type.value!r} is incompatible with expected_data_type "
                f"{expected_data_type!r} for target column {target_column_name!r}; "
                f"expected {data_type.parser_type.value!r}."
            )
        self._reject_keys(
            payload,
            self._parser_allowed_keys(parser_type),
            f"Parser for {target_column_name!r}",
        )

        # Resolve marker inheritance before validating ``replace_null_markers``. A column can use
        # global markers, replace them, or append its own markers while preserving first-seen order.
        markers_mode = self._enum_value(
            NullMarkersMode,
            payload.get("null_markers_mode", DEFAULT_NULL_MARKERS_MODE.value),
            "null_markers_mode",
        )
        if "null_markers_mode" in payload and "null_markers" not in payload:
            raise CompilationError(
                f"null_markers_mode for {target_column_name!r} requires column null_markers."
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
                f"replace_null_markers is true for {target_column_name!r}, "
                "but no null markers exist."
            )

        # Null defaults describe the final contract, not parse failure alone. They are permitted
        # only for non-nullable outputs and run after parsing and zero invalidation.
        is_nullable = self._bool(payload, "is_nullable", DEFAULT_IS_NULLABLE)
        raw_default_on_null = payload.get("default_on_null", _MISSING)
        if not is_nullable and raw_default_on_null is _MISSING:
            raise CompilationError(
                f"Column {target_column_name!r} is not nullable and requires default_on_null."
            )
        if is_nullable and raw_default_on_null is not _MISSING:
            raise CompilationError(
                f"default_on_null for {target_column_name!r} requires is_nullable: false."
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

        # Parse-error defaults are independent of null defaults: one handles invalid non-null
        # input; the other guarantees a final non-null value.
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
        # Preserving an invalid token is type-safe only when the target itself is a string. A raw
        # value such as ``Mul`` cannot inhabit an integer, date, or binary Spark column.
        # Enforcing this during compilation keeps runtime expressions schema-consistent and gives the
        # author a useful error before any Spark job starts.
        if on_parse_error is ParseErrorMode.PRESERVE and parser_type is not ParserType.STRING:
            raise CompilationError(
                f"on_parse_error: preserve for {target_column_name!r} requires a string parser."
            )
        raw_default_on_error = payload.get("default_on_error", _MISSING)
        if on_parse_error is ParseErrorMode.DEFAULT and raw_default_on_error is _MISSING:
            raise CompilationError(
                f"on_parse_error: default for {target_column_name!r} requires default_on_error."
            )
        if on_parse_error is not ParseErrorMode.DEFAULT and raw_default_on_error is not _MISSING:
            raise CompilationError(
                f"default_on_error for {target_column_name!r} requires on_parse_error: default."
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
        # Reject contradictory defaults at compile time. Otherwise the runtime would assign zero
        # and immediately invalidate it, producing a surprising null despite an authored default.
        if (
            parser_type in NUMERIC_PARSER_TYPES
            and not zero_is_valid
            and default_on_null is not None
            and Decimal(str(default_on_null)) == 0
        ):
            raise CompilationError(
                f"Column {target_column_name!r} rejects zero but uses zero as default_on_null."
            )
        if (
            parser_type in NUMERIC_PARSER_TYPES
            and not zero_is_valid
            and default_on_error is not None
            and Decimal(str(default_on_error)) == 0
        ):
            raise CompilationError(
                f"Column {target_column_name!r} rejects zero but uses zero as default_on_error."
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
            target_column_name,
            globals_config,
        )
        # Construct one fully resolved immutable options object only after every conditional
        # relationship has passed. Runtime code may safely branch on parser_type without looking
        # back at raw YAML.
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
            audit=self._bool(payload, "audit", DEFAULT_AUDIT),
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
        )
        self._validate_binary_defaults(compiled_options)
        return compiled_options

    def _validate_binary_defaults(
        self,
        options: ParserOptions,
    ) -> None:
        """Verify each authored binary default uses the parser's configured encoding."""
        if options.parser_type is not ParserType.BINARY:
            return
        for label, value in (
            ("default_on_null", options.default_on_null),
            ("default_on_error", options.default_on_error),
        ):
            if value is None:
                continue
            try:
                if options.binary_encoding is BinaryEncoding.BASE64:
                    base64.b64decode(value, validate=True)
                elif options.binary_encoding is BinaryEncoding.HEX:
                    # Spark's ``unhex`` accepts empty and odd-length hexadecimal strings (padding
                    # the leading nibble) but rejects whitespace. ``bytes.fromhex`` has the inverse
                    # edge behavior, so validate Spark's ASCII grammar directly.
                    if _ASCII_HEX_PATTERN.fullmatch(value) is None:
                        raise ValueError("not Spark hexadecimal text")
                else:
                    value.encode("utf-8")
            except (ValueError, UnicodeError, binascii.Error) as exc:
                raise CompilationError(
                    f"{label} is not valid {options.binary_encoding.value} binary text."
                ) from exc

    @staticmethod
    def _parser_allowed_keys(parser_type: ParserType) -> set[str]:
        """Return the exact common and parser-specific keys accepted for one scalar parser."""
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
            "audit",
        }
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
        }
        return common | specific[parser_type]

    def _compile_string_format(
        self,
        payload: Mapping[str, Any],
        parser_type: ParserType,
    ) -> StringFormat | None:
        """Resolve string formatting while leaving non-string parsers untouched."""
        if parser_type is not ParserType.STRING:
            return None
        value = payload.get("format")
        if value is None or (isinstance(value, str) and value.lower() in {"none", "null"}):
            return None
        return self._enum_value(StringFormat, value, "format")

    def _compile_formats(
        self,
        payload: Mapping[str, Any],
        parser_type: ParserType,
    ) -> tuple[str, ...]:
        """Resolve ordered datetime patterns for date and timestamp parser families."""
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
        if parser_type is ParserType.TIMESTAMP_NTZ:
            timezone_formats = [
                datetime_format
                for datetime_format in formats
                if _has_unquoted_timezone_pattern(datetime_format)
            ]
            if timezone_formats:
                raise CompilationError(
                    "formats for timestamp_ntz must not contain unquoted timezone or offset "
                    f"pattern fields: {timezone_formats}."
                )
        return formats

    def _compile_boolean_values(
        self,
        payload: Mapping[str, Any],
        parser_type: ParserType,
        target_column_name: str,
        globals_config: ParserGlobals,
    ) -> tuple[tuple[str, ...], tuple[str, ...], bool, BooleanValuesMode]:
        """Resolve inherited or extended Boolean tokens for one Boolean parser."""
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
                f"boolean_values_mode for {target_column_name!r} requires true_values "
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
            # Ordered de-duplication keeps serialized output deterministic and gives author-supplied tokens
            # predictable precedence in reports without changing membership behavior.
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
            f"target column {target_column_name!r}",
        )
        return true_values, false_values, case_sensitive, mode

    @staticmethod
    def _validate_boolean_overlap(
        true_values: tuple[str, ...],
        false_values: tuple[str, ...],
        case_sensitive: bool,
        label: str,
    ) -> None:
        """Reject overlap that is independent of the runtime's Unicode version.

        Exact strings always overlap. ASCII case conversion is also stable across Python and Spark.
        Non-ASCII case-insensitive overlap is deliberately deferred to a Spark expression so the
        compiler never rejects a valid vocabulary because its Unicode table differs from the JVM.
        """
        overlap = set(true_values) & set(false_values)
        if not case_sensitive:
            ascii_true = {item.lower() for item in true_values if item.isascii()}
            ascii_false = {item.lower() for item in false_values if item.isascii()}
            overlap.update(ascii_true & ascii_false)
        if overlap:
            raise CompilationError(
                f"Boolean true_values and false_values overlap for {label}: {sorted(overlap)}."
            )

    def _parser_type(self, value: str) -> ParserType:
        """Canonicalize aliases and convert a parser name into the closed enum vocabulary."""
        normalized = canonical_type_name(value)
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
        """Validate and normalize one scalar default value.

        Returned values use precise Python types such as ``Decimal``, ``date``, and ``datetime``.
        The runtime later turns this already-validated value into a native Spark literal.
        """
        if value is None:
            raise CompilationError(f"{label} must be non-null.")
        parser_type = data_type.parser_type
        if parser_type is ParserType.STRING:
            if not isinstance(value, str):
                raise CompilationError(f"{label} for string must be a string.")
            return _validate_utf8_string(value, f"{label} for string")
        if parser_type in {
            ParserType.BYTE,
            ParserType.SHORT,
            ParserType.INTEGER,
            ParserType.LONG,
        }:
            return self._integer_default(value, parser_type, label)
        if parser_type in {ParserType.FLOAT, ParserType.DOUBLE}:
            return self._floating_default(value, parser_type, label)
        if parser_type is ParserType.DECIMAL:
            return self._decimal_default(value, data_type, label)
        if parser_type is ParserType.BOOLEAN:
            if not isinstance(value, bool):
                raise CompilationError(f"{label} for boolean must be true or false.")
            return value
        if parser_type is ParserType.DATE:
            return self._date_default(value, label)
        if parser_type in {ParserType.TIMESTAMP, ParserType.TIMESTAMP_NTZ}:
            return self._timestamp_default(value, parser_type, label)
        if parser_type is ParserType.BINARY:
            if not isinstance(value, str):
                raise CompilationError(f"{label} for binary must be an encoded string.")
            return _validate_utf8_string(value, f"{label} for binary")
        raise CompilationError(f"Unsupported parser type for {label}: {parser_type.value}.")

    @staticmethod
    def _integer_default(value: Any, parser_type: ParserType, label: str) -> int:
        """Validate an exact Python integer against the selected Spark signed range."""
        # ``bool`` subclasses ``int`` in Python, so it must be excluded explicitly.
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
    def _floating_default(value: Any, parser_type: ParserType, label: str) -> float:
        """Validate a finite default that survives conversion to its declared Spark width.

        Python's ``float`` and Spark ``double`` are binary64 values, but Spark ``float`` is binary32.
        Checking only ``math.isfinite`` would allow a finite Python value to become infinity—or a
        tiny non-zero value to become zero—when Spark narrows it. The standard-library round trip
        below exactly models that final binary32 representation before the config is accepted.
        """
        if isinstance(value, bool) or not isinstance(value, (int, float, Decimal)):
            raise CompilationError(f"{label} for {parser_type.value} must be numeric.")
        try:
            converted = float(value)
        except (OverflowError, ValueError) as exc:
            raise CompilationError(f"{label} for {parser_type.value} must be finite.") from exc
        if not math.isfinite(converted):
            raise CompilationError(f"{label} for {parser_type.value} must be finite.")
        if converted == 0 and value != 0:
            raise CompilationError(f"{label} for {parser_type.value} underflows to zero in Spark.")
        if parser_type is ParserType.FLOAT:
            try:
                narrowed = struct.unpack("!f", struct.pack("!f", converted))[0]
            except OverflowError as exc:
                raise CompilationError(
                    f"{label} for float is outside Spark's finite float32 range."
                ) from exc
            if not math.isfinite(narrowed):
                raise CompilationError(
                    f"{label} for float is outside Spark's finite float32 range."
                )
            if converted != 0 and narrowed == 0:
                raise CompilationError(f"{label} for float underflows to zero in Spark float32.")
        return converted

    @staticmethod
    def _decimal_default(value: Any, data_type: SparkDataType, label: str) -> Decimal:
        """Validate and canonically quantize an exact decimal without binary floating point."""
        if isinstance(value, str):
            _validate_utf8_string(value, f"{label} for decimal")
            if _DECIMAL_DEFAULT_PATTERN.fullmatch(value) is None:
                raise CompilationError(f"{label} for decimal must be numeric.")
        try:
            converted = Decimal(str(value))
        except (InvalidOperation, ValueError) as exc:
            raise CompilationError(f"{label} for decimal must be numeric.") from exc
        if not converted.is_finite():
            raise CompilationError(f"{label} for decimal must be finite.")
        precision = data_type.precision
        scale = data_type.scale
        assert precision is not None and scale is not None
        sign, digit_tuple, exponent = converted.as_tuple()
        if not isinstance(exponent, int):
            raise CompilationError(f"{label} for decimal must be finite.")
        # Canonicalize every spelling of zero (including negative and scientific zero) to one
        # positive value with the declared scale. This makes serialization and content hashing
        # independent of authoring notation.
        if not converted:
            return Decimal((0, (0,), -scale))

        # Strip insignificant trailing zeroes before checking scale. Values such as ``1.2300`` fit
        # decimal(*,2) exactly and should canonicalize to the same value as ``1.23``.
        digits = list(digit_tuple)
        while digits[-1] == 0:
            digits.pop()
            exponent += 1

        # Decimal's tuple representation lets us validate range and exact scale without invoking a
        # context-limited quantize operation. Non-zero digits beyond the declared scale are rejected
        # rather than rounded.
        integral_digits = max(len(digits) + exponent, 0)
        if integral_digits > precision - scale or exponent < -scale:
            raise CompilationError(f"{label} does not fit {data_type.canonical}.")
        quantized_digits = tuple(digits) + (0,) * (exponent + scale)
        return Decimal((sign, quantized_digits, -scale))

    @staticmethod
    def _date_default(value: Any, label: str) -> date:
        """Accept a date object or strict ISO date string, never a datetime."""
        if isinstance(value, datetime):
            raise CompilationError(f"{label} for date must not contain a time component.")
        if isinstance(value, date):
            return date(value.year, value.month, value.day)
        if isinstance(value, str):
            _validate_utf8_string(value, f"{label} for date")
            if _DATE_DEFAULT_PATTERN.fullmatch(value) is None:
                raise CompilationError(f"{label} for date must use ISO YYYY-MM-DD.")
            try:
                return date.fromisoformat(value)
            except ValueError as exc:
                raise CompilationError(f"{label} for date must use ISO YYYY-MM-DD.") from exc
        raise CompilationError(f"{label} for date must be a date or ISO string.")

    @staticmethod
    def _timestamp_default(value: Any, parser_type: ParserType, label: str) -> datetime:
        """Accept an ISO datetime and enforce timestamp-vs-wall-clock timezone semantics."""
        if isinstance(value, datetime):
            parsed = value
        elif isinstance(value, str):
            _validate_utf8_string(value, f"{label} for timestamp")
            if _TIMESTAMP_DEFAULT_PATTERN.fullmatch(value) is None:
                raise CompilationError(
                    f"{label} for timestamp must use ISO YYYY-MM-DDTHH:MM:SS[.ffffff][Z|+HH:MM]."
                )
            try:
                # Python 3.10's ``fromisoformat`` accepts only three or six fractional digits,
                # while the public grammar deliberately accepts one through six. Canonicalize the
                # already-validated fraction so every supported interpreter compiles identically.
                normalized = re.sub(
                    r"\.([0-9]{1,6})(?=Z|[+-][0-9]{2}:[0-9]{2}|\Z)",
                    lambda match: f".{match.group(1).ljust(6, '0')}",
                    value,
                )
                parsed = datetime.fromisoformat(
                    f"{normalized[:-1]}+00:00" if normalized.endswith("Z") else normalized
                )
            except ValueError as exc:
                raise CompilationError(f"{label} for timestamp must be an ISO timestamp.") from exc
        else:
            raise CompilationError(f"{label} for timestamp must be a datetime or ISO string.")
        try:
            offset = parsed.utcoffset()
        except (OverflowError, TypeError, ValueError) as exc:
            raise CompilationError(
                f"{label} for timestamp has an invalid timezone offset."
            ) from exc
        if parser_type is ParserType.TIMESTAMP_NTZ and offset is not None:
            raise CompilationError(
                f"{label} for timestamp_ntz must not include a timezone offset; "
                "timestamp_ntz represents a local wall-clock value."
            )
        if offset is not None and offset.total_seconds() % 60:
            raise CompilationError(
                f"{label} for timestamp must use a whole-minute timezone offset."
            )
        # Rebuild a base ``datetime`` so subclass state cannot mutate compiled behavior. Spark
        # TimestampType stores an instant: PySpark converts every aware datetime through its UTC
        # timetable. Canonicalizing the same way ensures equivalent offsets share serialization and
        # content hashes rather than preserving a behaviorally irrelevant spelling difference.
        canonical = datetime(
            parsed.year,
            parsed.month,
            parsed.day,
            parsed.hour,
            parsed.minute,
            parsed.second,
            parsed.microsecond,
            tzinfo=timezone.utc,
            fold=parsed.fold,
        )
        if offset is None:
            return canonical.replace(tzinfo=None)
        try:
            canonical -= offset
        except OverflowError as exc:
            raise CompilationError(
                f"{label} for timestamp is outside the supported range after UTC normalization."
            ) from exc
        return canonical

    def _validate_unique_columns(self, columns: tuple[ColumnParser, ...]) -> None:
        """Reject duplicate target names while intentionally allowing repeated sources."""
        column_names = [column.target_column_name for column in columns]
        duplicate_columns = self._duplicates(column_names)
        if duplicate_columns:
            raise CompilationError(f"Duplicate target_column_name values: {duplicate_columns}.")

    def _string_sequence(
        self,
        value: Any,
        label: str,
        *,
        allow_empty_values: bool = True,
    ) -> tuple[str, ...]:
        """Validate a YAML string list and remove duplicates without reordering it."""
        if not isinstance(value, list):
            raise CompilationError(f"{label} must be a YAML list of strings.")
        invalid = [
            item
            for item in value
            if not isinstance(item, str) or (not allow_empty_values and not item)
        ]
        if invalid:
            raise CompilationError(f"{label} must contain only valid strings: {invalid!r}.")
        for index, item in enumerate(value):
            _validate_utf8_string(item, f"{label}[{index}]")
        return self._deduplicate(tuple(value))

    @staticmethod
    def _deduplicate(values: tuple[str, ...]) -> tuple[str, ...]:
        """Return first-occurrence-preserving unique strings."""
        return tuple(dict.fromkeys(values))

    @staticmethod
    def _duplicates(values: list[str]) -> list[str]:
        """Return sorted duplicate strings in linear scan time."""
        seen: set[str] = set()
        duplicates: set[str] = set()
        for value in values:
            if value in seen:
                duplicates.add(value)
            else:
                seen.add(value)
        return sorted(duplicates)

    def _enum_value(self, enum_type: type[_EnumT], value: Any, label: str) -> _EnumT:
        """Parse one case-insensitive enum value with an actionable allowed-values error."""
        if not isinstance(value, str):
            raise CompilationError(f"{label} must be a string.")
        _validate_utf8_string(value, label)
        try:
            return enum_type(value.lower())
        except ValueError as exc:
            valid = ", ".join(member.value for member in enum_type)
            raise CompilationError(f"Invalid {label} {value!r}. Valid values: {valid}.") from exc

    @staticmethod
    def _bool(payload: Mapping[str, Any], key: str, default: bool) -> bool:
        """Read a strict YAML Boolean; strings such as ``"true"`` are not accepted."""
        if key not in payload:
            return default
        value = payload[key]
        if not isinstance(value, bool):
            raise CompilationError(f"{key} must be true or false.")
        return value

    @staticmethod
    def _required_string(payload: Mapping[str, Any], key: str) -> str:
        """Read and trim a required non-empty string."""
        value = payload.get(key)
        if not isinstance(value, str) or not value.strip():
            raise CompilationError(f"{key} must be a non-empty string.")
        _validate_utf8_string(value, key)
        return value.strip()

    @staticmethod
    def _required_identifier(payload: Mapping[str, Any], key: str) -> str:
        """Read a non-blank source/target name without changing its authored identity."""
        value = payload.get(key)
        if not isinstance(value, str) or not value.strip():
            raise CompilationError(f"{key} must be a non-empty string.")
        return _validate_utf8_string(value, key)

    @staticmethod
    def _optional_string(payload: Mapping[str, Any], key: str) -> str | None:
        """Read an optional string while preserving intentional surrounding text."""
        value = payload.get(key)
        if value is None:
            return None
        if not isinstance(value, str):
            raise CompilationError(f"{key} must be a string when provided.")
        return _validate_utf8_string(value, key)

    @staticmethod
    def _ensure_mapping(value: Any, label: str) -> Mapping[str, Any]:
        """Require a mapping with string keys before key-set validation begins."""
        if not isinstance(value, Mapping):
            raise CompilationError(f"{label} must be a mapping.")
        invalid_keys = [key for key in value if not isinstance(key, str)]
        if invalid_keys:
            raise CompilationError(f"{label} keys must be strings: {invalid_keys!r}.")
        for key in value:
            _validate_utf8_string(key, f"{label} key")
        return value

    @staticmethod
    def _reject_keys(payload: Mapping[str, Any], allowed: set[str], label: str) -> None:
        """Fail closed on misspelled or unsupported authoring keys."""
        unsupported = sorted(set(payload) - allowed)
        if unsupported:
            raise CompilationError(f"{label} contains unsupported keys: {unsupported}.")
