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
import reprlib
import struct
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import date, datetime, timezone
from decimal import Decimal, InvalidOperation
from enum import Enum
from functools import partial
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


def _recursion_safe_repr(value: Any) -> str:
    """Preserve ordinary diagnostic text while bounding pathological recursive rendering."""
    try:
        return repr(value)
    except RecursionError:
        return reprlib.repr(value)


@dataclass(frozen=True)
class _ValidationIssue:
    """One semantic authoring problem and the path that identifies its input location."""

    path: str
    message: str

    def render(self) -> str:
        """Render an unambiguous message for a multi-error result."""
        return f"{self.path}: {self.message}" if self.path else self.message


class _IssueCollector:
    """Collect independent compiler failures without constructing a partial public config."""

    def __init__(self) -> None:
        self.issues: list[_ValidationIssue] = []
        self._seen: set[_ValidationIssue] = set()

    def add(self, path: str, message: str) -> None:
        """Append one issue in validation order."""
        issue = _ValidationIssue(path, message)
        if issue in self._seen:
            return
        self._seen.add(issue)
        self.issues.append(issue)

    def capture(self, path: str, operation: Callable[[], Any]) -> Any:
        """Run one existing validator, recording its public error and returning a sentinel."""
        try:
            return operation()
        except CompilationError as exc:
            for error in exc.errors:
                self.add(path, error)
            return _MISSING

    def raise_if_any(self) -> None:
        """Raise once, preserving legacy text when exactly one issue was found."""
        if not self.issues:
            return
        errors = (
            (self.issues[0].message,)
            if len(self.issues) == 1
            else tuple(issue.render() for issue in self.issues)
        )
        raise CompilationError(errors)


@dataclass(frozen=True)
class _GlobalsValidation:
    """Usable global fallbacks plus provenance needed to suppress cascading column errors."""

    config: ParserGlobals
    null_markers_valid: bool = True
    true_values_valid: bool = True
    false_values_valid: bool = True
    boolean_case_sensitive_valid: bool = True
    boolean_overlap_values: frozenset[str] = frozenset()


@dataclass(frozen=True)
class _NullMarkerValidation:
    """Resolved marker options produced during dependency-aware parser validation."""

    markers: tuple[str, ...]
    mode: NullMarkersMode
    replace: bool
    case_sensitive: bool


@dataclass(frozen=True)
class _NullDefaultValidation:
    """Resolved final-null policy and whether its typed value can be checked further."""

    is_nullable: bool
    value: Any
    value_valid: bool


@dataclass(frozen=True)
class _ParseErrorValidation:
    """Resolved parse-error policy and whether its typed value can be checked further."""

    mode: ParseErrorMode
    value: Any
    value_valid: bool


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

        Once the outer mapping is trustworthy, independent semantic issues are collected in a
        deterministic validation order. Dependent checks run only when their prerequisites are
        valid, so the aggregate does not contain misleading cascade errors. No partial public model
        escapes on failure.
        """
        payload = self._ensure_mapping(payload, "parser config")
        collector = _IssueCollector()
        collector.capture(
            "parser config",
            lambda: self._reject_keys(
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
            ),
        )

        parser_config_id = collector.capture(
            "parser_config_id", lambda: self._required_string(payload, "parser_config_id")
        )
        parser_config_name = collector.capture(
            "parser_config_name", lambda: self._required_string(payload, "parser_config_name")
        )
        version = collector.capture("version", lambda: self._required_string(payload, "version"))
        description = collector.capture(
            "description", lambda: self._optional_string(payload, "description")
        )
        owner = collector.capture("owner", lambda: self._optional_string(payload, "owner"))
        owner_department = collector.capture(
            "owner_department", lambda: self._optional_string(payload, "owner_department")
        )

        # Globals must be compiled before columns because column vocabularies can inherit or extend
        # them. Invalid authored values use internal defaults only to let unrelated column checks
        # continue; the collector still prevents those fallbacks from becoming a public config.
        globals_validation = self._compile_globals(payload.get("globals", {}), collector)
        raw_columns = payload.get("columns")
        columns: list[ColumnParser] = []
        target_names: list[str] = []
        if not isinstance(raw_columns, list) or not raw_columns:
            collector.add("columns", "columns must be a non-empty list.")
        else:
            for index, raw_column in enumerate(raw_columns, start=1):
                column, target_name = self._compile_column(
                    raw_column,
                    globals_validation,
                    index,
                    collector,
                )
                if column is not None:
                    columns.append(column)
                if target_name is not None:
                    target_names.append(target_name)
            duplicate_columns = self._duplicates(target_names)
            if duplicate_columns:
                collector.add(
                    "columns",
                    f"Duplicate target_column_name values: {duplicate_columns}.",
                )

        collector.raise_if_any()
        assert all(
            value is not _MISSING
            for value in (
                parser_config_id,
                parser_config_name,
                version,
                description,
                owner,
                owner_department,
            )
        )
        return ParserConfig(
            parser_config_id=parser_config_id,
            parser_config_name=parser_config_name,
            version=version,
            columns=tuple(columns),
            globals=globals_validation.config,
            description=description,
            owner=owner,
            owner_department=owner_department,
        )

    def _compile_globals(
        self,
        raw_globals: Any,
        collector: _IssueCollector,
    ) -> _GlobalsValidation:
        """Validate globals independently and retain safe fallbacks for later column checks."""
        payload = collector.capture("globals", lambda: self._ensure_mapping(raw_globals, "globals"))
        if payload is _MISSING:
            return _GlobalsValidation(
                ParserGlobals(),
                null_markers_valid=False,
                true_values_valid=False,
                false_values_valid=False,
                boolean_case_sensitive_valid=False,
                boolean_overlap_values=frozenset(),
            )

        collector.capture(
            "globals",
            lambda: self._reject_keys(
                payload,
                {
                    "null_markers",
                    "null_marker_case_sensitive",
                    "true_values",
                    "false_values",
                    "boolean_case_sensitive",
                },
                "globals",
            ),
        )
        true_values_result = collector.capture(
            "globals.true_values",
            lambda: self._string_sequence(
                payload.get("true_values", list(DEFAULT_BOOLEAN_TRUE_VALUES)),
                "globals.true_values",
                allow_empty_values=False,
            ),
        )
        false_values_result = collector.capture(
            "globals.false_values",
            lambda: self._string_sequence(
                payload.get("false_values", list(DEFAULT_BOOLEAN_FALSE_VALUES)),
                "globals.false_values",
                allow_empty_values=False,
            ),
        )
        true_values_valid = true_values_result is not _MISSING
        false_values_valid = false_values_result is not _MISSING
        true_values = true_values_result if true_values_valid else DEFAULT_BOOLEAN_TRUE_VALUES
        false_values = false_values_result if false_values_valid else DEFAULT_BOOLEAN_FALSE_VALUES
        vocabularies_nonempty = not (
            (true_values_valid and not true_values) or (false_values_valid and not false_values)
        )
        if not vocabularies_nonempty:
            collector.add(
                "globals",
                "globals.true_values and globals.false_values must be non-empty.",
            )

        boolean_case_sensitive_result = collector.capture(
            "globals.boolean_case_sensitive",
            lambda: self._bool(
                payload,
                "boolean_case_sensitive",
                DEFAULT_BOOLEAN_CASE_SENSITIVE,
            ),
        )
        boolean_case_sensitive_valid = boolean_case_sensitive_result is not _MISSING
        boolean_case_sensitive = (
            boolean_case_sensitive_result
            if boolean_case_sensitive_valid
            else DEFAULT_BOOLEAN_CASE_SENSITIVE
        )
        # Exact overlap is independent of case sensitivity. If that flag is invalid, validate the
        # exact sets now and defer only the case-folded comparison.
        boolean_overlap_values: frozenset[str] = frozenset()
        if true_values_valid and false_values_valid and vocabularies_nonempty:
            boolean_overlap_values = frozenset(
                self._boolean_overlap_values(
                    true_values,
                    false_values,
                    boolean_case_sensitive if boolean_case_sensitive_valid else True,
                )
            )
            collector.capture(
                "globals",
                lambda: self._validate_boolean_overlap(
                    true_values,
                    false_values,
                    (boolean_case_sensitive if boolean_case_sensitive_valid else True),
                    "globals",
                ),
            )

        null_markers_result = collector.capture(
            "globals.null_markers",
            lambda: self._string_sequence(
                payload.get("null_markers", list(DEFAULT_NULL_MARKERS)),
                "null_markers",
            ),
        )
        null_markers_valid = null_markers_result is not _MISSING
        null_markers = null_markers_result if null_markers_valid else DEFAULT_NULL_MARKERS
        null_marker_case_sensitive_result = collector.capture(
            "globals.null_marker_case_sensitive",
            lambda: self._bool(
                payload,
                "null_marker_case_sensitive",
                DEFAULT_NULL_MARKER_CASE_SENSITIVE,
            ),
        )
        null_marker_case_sensitive = (
            null_marker_case_sensitive_result
            if null_marker_case_sensitive_result is not _MISSING
            else DEFAULT_NULL_MARKER_CASE_SENSITIVE
        )
        return _GlobalsValidation(
            ParserGlobals(
                null_markers=null_markers,
                null_marker_case_sensitive=null_marker_case_sensitive,
                true_values=true_values,
                false_values=false_values,
                boolean_case_sensitive=boolean_case_sensitive,
            ),
            null_markers_valid=null_markers_valid,
            true_values_valid=true_values_valid and bool(true_values),
            false_values_valid=false_values_valid and bool(false_values),
            boolean_case_sensitive_valid=boolean_case_sensitive_valid,
            boolean_overlap_values=boolean_overlap_values,
        )

    def _compile_column(
        self,
        raw_column: Any,
        globals_validation: _GlobalsValidation,
        index: int,
        collector: _IssueCollector,
    ) -> tuple[ColumnParser | None, str | None]:
        """Validate one binding, returning its valid target even when another field fails."""
        path = f"columns[{index - 1}]"
        issue_count = len(collector.issues)
        payload = collector.capture(
            path,
            lambda: self._ensure_mapping(raw_column, f"column at index {index}"),
        )
        if payload is _MISSING:
            return None, None
        collector.capture(
            path,
            lambda: self._reject_keys(
                payload,
                {
                    "source_column_name",
                    "target_column_name",
                    "expected_data_type",
                    "parser",
                },
                f"Column at index {index}",
            ),
        )
        source_column_name = collector.capture(
            f"{path}.source_column_name",
            lambda: self._required_identifier(payload, "source_column_name"),
        )
        target_column_name = collector.capture(
            f"{path}.target_column_name",
            lambda: self._required_identifier(payload, "target_column_name"),
        )
        valid_target_name = target_column_name if target_column_name is not _MISSING else None
        # Parse the scalar DDL once and carry both its model and canonical text. Complex DDL is
        # rejected by ``parse_spark_data_type`` before parser options are considered.
        data_type = collector.capture(
            f"{path}.expected_data_type",
            lambda: parse_spark_data_type(self._required_string(payload, "expected_data_type")),
        )
        options = self._compile_parser(
            payload.get("parser", _MISSING),
            globals_validation,
            None if data_type is _MISSING else data_type,
            (
                target_column_name
                if target_column_name is not _MISSING
                else f"column at index {index}"
            ),
            f"{path}.parser",
            collector,
        )
        if len(collector.issues) != issue_count:
            return None, valid_target_name
        assert source_column_name is not _MISSING
        assert target_column_name is not _MISSING
        assert data_type is not _MISSING
        assert options is not None
        return ColumnParser(
            source_column_name=source_column_name,
            target_column_name=target_column_name,
            expected_data_type=data_type.canonical,
            data_type=data_type,
            parser=options,
        ), valid_target_name

    def _compile_parser(
        self,
        raw_parser: Any,
        globals_validation: _GlobalsValidation,
        data_type: SparkDataType | None,
        target_column_name: str,
        path: str,
        collector: _IssueCollector,
    ) -> ParserOptions | None:
        """Resolve one scalar parser while collecting independent option failures."""
        issue_count = len(collector.issues)
        if raw_parser is _MISSING:
            collector.add(path, f"parser is required for column {target_column_name!r}.")
            return None
        # Scalar shorthand such as ``parser: date`` is normalized into the same mapping path as
        # long-form options. From this point onward there is only one validation implementation.
        if isinstance(raw_parser, str):
            payload: Mapping[str, Any] = {"type": raw_parser}
        else:
            payload = collector.capture(
                path,
                lambda: self._ensure_mapping(raw_parser, f"parser for {target_column_name!r}"),
            )
            if payload is _MISSING:
                return None

        parser_type_result = collector.capture(
            f"{path}.type",
            lambda: self._parser_type(self._required_string(payload, "type")),
        )
        parser_type = None if parser_type_result is _MISSING else parser_type_result
        types_compatible = (
            parser_type is not None
            and data_type is not None
            and parser_type is data_type.parser_type
        )
        if parser_type is not None and data_type is not None and not types_compatible:
            collector.add(
                f"{path}.type",
                f"Parser {parser_type.value!r} is incompatible with expected_data_type "
                f"{data_type.canonical!r} for target column {target_column_name!r}; "
                f"expected {data_type.parser_type.value!r}.",
            )
        allowed_keys = (
            self._parser_allowed_keys(parser_type)
            if parser_type is not None
            else set().union(*(self._parser_allowed_keys(member) for member in ParserType))
        )
        collector.capture(
            path,
            lambda: self._reject_keys(
                payload,
                allowed_keys,
                f"Parser for {target_column_name!r}",
            ),
        )

        marker_options = self._compile_null_marker_options(
            payload,
            globals_validation,
            target_column_name,
            path,
            collector,
        )

        # Null defaults describe the final contract, not parse failure alone. They are permitted
        # only for non-nullable outputs and run after parsing and zero invalidation.
        null_default = self._compile_null_default(
            payload,
            data_type if types_compatible else None,
            target_column_name,
            path,
            collector,
        )

        # Parse-error defaults are independent of null defaults: one handles invalid non-null
        # input; the other guarantees a final non-null value.
        parse_error = self._compile_parse_error_default(
            payload,
            parser_type,
            data_type if types_compatible else None,
            target_column_name,
            path,
            collector,
        )
        zero_is_valid = self._compile_zero_policy(
            payload,
            parser_type,
            types_compatible,
            null_default,
            parse_error,
            target_column_name,
            path,
            collector,
        )

        string_format_result = (
            collector.capture(
                f"{path}.format",
                lambda: self._compile_string_format(payload, parser_type),
            )
            if parser_type is ParserType.STRING
            else None
        )
        string_format = None if string_format_result is _MISSING else string_format_result
        formats_result = (
            collector.capture(
                f"{path}.formats",
                lambda: self._compile_formats(payload, parser_type),
            )
            if parser_type in {ParserType.DATE, ParserType.TIMESTAMP, ParserType.TIMESTAMP_NTZ}
            else ()
        )
        formats = () if formats_result is _MISSING else formats_result
        (
            true_values,
            false_values,
            boolean_case_sensitive,
            boolean_values_mode,
        ) = self._compile_boolean_values(
            payload,
            parser_type,
            target_column_name,
            globals_validation,
            path,
            collector,
        )

        def captured_bool(key: str, default: bool) -> bool:
            result = collector.capture(f"{path}.{key}", lambda: self._bool(payload, key, default))
            return default if result is _MISSING else result

        trim_whitespace = captured_bool("trim_whitespace", DEFAULT_TRIM_WHITESPACE)
        collapse_whitespace = captured_bool("collapse_whitespace", DEFAULT_COLLAPSE_WHITESPACE)
        empty_is_null = captured_bool("empty_is_null", DEFAULT_EMPTY_IS_NULL)
        audit = captured_bool("audit", DEFAULT_AUDIT)

        if parser_type is ParserType.BINARY:
            binary_encoding_result = collector.capture(
                f"{path}.encoding",
                lambda: self._enum_value(
                    BinaryEncoding,
                    payload.get("encoding", DEFAULT_BINARY_ENCODING.value),
                    "encoding",
                ),
            )
            binary_encoding = (
                binary_encoding_result
                if binary_encoding_result is not _MISSING
                else DEFAULT_BINARY_ENCODING
            )
            if binary_encoding_result is not _MISSING and types_compatible:
                for label, value, valid in (
                    ("default_on_null", null_default.value, null_default.value_valid),
                    ("default_on_error", parse_error.value, parse_error.value_valid),
                ):
                    if valid and value is not None:
                        collector.capture(
                            f"{path}.{label}",
                            partial(
                                self._validate_binary_default,
                                value,
                                binary_encoding,
                                label,
                            ),
                        )
        else:
            binary_encoding = DEFAULT_BINARY_ENCODING

        if len(collector.issues) != issue_count or parser_type is None:
            return None
        # Construct one fully resolved immutable options object only after every conditional
        # relationship has passed. Runtime code may safely branch on parser_type without looking
        # back at raw YAML.
        return ParserOptions(
            parser_type=parser_type,
            trim_whitespace=trim_whitespace,
            collapse_whitespace=collapse_whitespace,
            empty_is_null=empty_is_null,
            replace_null_markers=marker_options.replace,
            null_markers=marker_options.markers,
            null_markers_mode=marker_options.mode,
            null_marker_case_sensitive=marker_options.case_sensitive,
            is_nullable=null_default.is_nullable,
            default_on_null=null_default.value,
            on_parse_error=parse_error.mode,
            default_on_error=parse_error.value,
            audit=audit,
            zero_is_valid=zero_is_valid,
            string_format=string_format,
            formats=formats,
            true_values=true_values,
            false_values=false_values,
            boolean_case_sensitive=boolean_case_sensitive,
            boolean_values_mode=boolean_values_mode,
            binary_encoding=binary_encoding,
        )

    def _compile_null_marker_options(
        self,
        payload: Mapping[str, Any],
        globals_validation: _GlobalsValidation,
        target_column_name: str,
        path: str,
        collector: _IssueCollector,
    ) -> _NullMarkerValidation:
        """Resolve marker inheritance and collect independent marker-option issues."""
        mode_result = collector.capture(
            f"{path}.null_markers_mode",
            lambda: self._enum_value(
                NullMarkersMode,
                payload.get("null_markers_mode", DEFAULT_NULL_MARKERS_MODE.value),
                "null_markers_mode",
            ),
        )
        mode_valid = mode_result is not _MISSING
        mode = mode_result if mode_valid else DEFAULT_NULL_MARKERS_MODE
        markers_requirement_valid = not (
            "null_markers_mode" in payload and "null_markers" not in payload
        )
        if not markers_requirement_valid:
            collector.add(
                f"{path}.null_markers",
                f"null_markers_mode for {target_column_name!r} requires column null_markers.",
            )

        column_markers: tuple[str, ...] = ()
        column_markers_valid = True
        if "null_markers" in payload:
            column_markers_result = collector.capture(
                f"{path}.null_markers",
                lambda: self._string_sequence(payload["null_markers"], "null_markers"),
            )
            column_markers_valid = column_markers_result is not _MISSING
            column_markers = column_markers_result if column_markers_valid else ()
            markers_valid = column_markers_valid and mode_valid
            if mode is NullMarkersMode.EXTEND:
                markers_valid = markers_valid and globals_validation.null_markers_valid
                markers = self._deduplicate(
                    (*globals_validation.config.null_markers, *column_markers)
                )
            else:
                markers = column_markers
        else:
            markers = globals_validation.config.null_markers
            markers_valid = globals_validation.null_markers_valid and mode_valid

        replace_result = collector.capture(
            f"{path}.replace_null_markers",
            lambda: self._bool(
                payload,
                "replace_null_markers",
                DEFAULT_REPLACE_NULL_MARKERS,
            ),
        )
        replace = replace_result if replace_result is not _MISSING else DEFAULT_REPLACE_NULL_MARKERS
        markers_guaranteed_empty = markers_valid and not markers
        if (
            not mode_valid
            and "null_markers" in payload
            and column_markers_valid
            and not column_markers
            and globals_validation.null_markers_valid
            and not globals_validation.config.null_markers
        ):
            # Both valid mode interpretations resolve to an empty vocabulary, so this relationship
            # can still be diagnosed without guessing which mode the author intended.
            markers_guaranteed_empty = True
        if (
            replace_result is not _MISSING
            and replace
            and markers_requirement_valid
            and markers_guaranteed_empty
        ):
            collector.add(
                f"{path}.replace_null_markers",
                f"replace_null_markers is true for {target_column_name!r}, "
                "but no null markers exist.",
            )

        case_sensitive_result = collector.capture(
            f"{path}.null_marker_case_sensitive",
            lambda: self._bool(
                payload,
                "null_marker_case_sensitive",
                globals_validation.config.null_marker_case_sensitive,
            ),
        )
        case_sensitive = (
            case_sensitive_result
            if case_sensitive_result is not _MISSING
            else globals_validation.config.null_marker_case_sensitive
        )
        return _NullMarkerValidation(markers, mode, replace, case_sensitive)

    def _compile_null_default(
        self,
        payload: Mapping[str, Any],
        data_type: SparkDataType | None,
        target_column_name: str,
        path: str,
        collector: _IssueCollector,
    ) -> _NullDefaultValidation:
        """Resolve final-null behavior, skipping typed checks when its contract is invalid."""
        is_nullable_result = collector.capture(
            f"{path}.is_nullable",
            lambda: self._bool(payload, "is_nullable", DEFAULT_IS_NULLABLE),
        )
        is_nullable = (
            is_nullable_result if is_nullable_result is not _MISSING else DEFAULT_IS_NULLABLE
        )
        raw_default = payload.get("default_on_null", _MISSING)
        if is_nullable_result is _MISSING:
            return _NullDefaultValidation(is_nullable, None, raw_default is _MISSING)
        if not is_nullable and raw_default is _MISSING:
            collector.add(
                f"{path}.default_on_null",
                f"Column {target_column_name!r} is not nullable and requires default_on_null.",
            )
            return _NullDefaultValidation(is_nullable, None, False)
        if is_nullable and raw_default is not _MISSING:
            collector.add(
                f"{path}.default_on_null",
                f"default_on_null for {target_column_name!r} requires is_nullable: false.",
            )
            return _NullDefaultValidation(is_nullable, None, False)
        if raw_default is _MISSING:
            return _NullDefaultValidation(is_nullable, None, True)
        if data_type is None:
            return _NullDefaultValidation(is_nullable, None, False)

        default_result = collector.capture(
            f"{path}.default_on_null",
            lambda: self._typed_default(raw_default, data_type, "default_on_null"),
        )
        return _NullDefaultValidation(
            is_nullable,
            None if default_result is _MISSING else default_result,
            default_result is not _MISSING,
        )

    def _compile_parse_error_default(
        self,
        payload: Mapping[str, Any],
        parser_type: ParserType | None,
        data_type: SparkDataType | None,
        target_column_name: str,
        path: str,
        collector: _IssueCollector,
    ) -> _ParseErrorValidation:
        """Resolve parse-error behavior and its optional typed default."""
        raw_mode = payload.get("on_parse_error", DEFAULT_ON_PARSE_ERROR.value)
        # YAML resolves an unquoted ``null`` scalar to ``None``. Here it names the null mode.
        if "on_parse_error" in payload and raw_mode is None:
            raw_mode = ParseErrorMode.NULL.value
        mode_result = collector.capture(
            f"{path}.on_parse_error",
            lambda: self._enum_value(ParseErrorMode, raw_mode, "on_parse_error"),
        )
        mode = mode_result if mode_result is not _MISSING else DEFAULT_ON_PARSE_ERROR
        if (
            mode_result is not _MISSING
            and mode is ParseErrorMode.PRESERVE
            and parser_type is not None
            and parser_type is not ParserType.STRING
        ):
            collector.add(
                f"{path}.on_parse_error",
                f"on_parse_error: preserve for {target_column_name!r} requires a string parser.",
            )

        raw_default = payload.get("default_on_error", _MISSING)
        if mode_result is _MISSING:
            return _ParseErrorValidation(mode, None, raw_default is _MISSING)
        if mode is ParseErrorMode.DEFAULT and raw_default is _MISSING:
            collector.add(
                f"{path}.default_on_error",
                f"on_parse_error: default for {target_column_name!r} requires default_on_error.",
            )
            return _ParseErrorValidation(mode, None, False)
        if mode is not ParseErrorMode.DEFAULT and raw_default is not _MISSING:
            collector.add(
                f"{path}.default_on_error",
                f"default_on_error for {target_column_name!r} requires on_parse_error: default.",
            )
            return _ParseErrorValidation(mode, None, False)
        if raw_default is _MISSING:
            return _ParseErrorValidation(mode, None, True)
        if data_type is None:
            return _ParseErrorValidation(mode, None, False)

        default_result = collector.capture(
            f"{path}.default_on_error",
            lambda: self._typed_default(raw_default, data_type, "default_on_error"),
        )
        return _ParseErrorValidation(
            mode,
            None if default_result is _MISSING else default_result,
            default_result is not _MISSING,
        )

    def _compile_zero_policy(
        self,
        payload: Mapping[str, Any],
        parser_type: ParserType | None,
        types_compatible: bool,
        null_default: _NullDefaultValidation,
        parse_error: _ParseErrorValidation,
        target_column_name: str,
        path: str,
        collector: _IssueCollector,
    ) -> bool:
        """Resolve numeric zero validity and reject contradictory validated defaults."""
        if parser_type not in NUMERIC_PARSER_TYPES:
            return DEFAULT_ZERO_IS_VALID
        zero_result = collector.capture(
            f"{path}.zero_is_valid",
            lambda: self._bool(payload, "zero_is_valid", DEFAULT_ZERO_IS_VALID),
        )
        zero_is_valid = zero_result if zero_result is not _MISSING else DEFAULT_ZERO_IS_VALID
        if zero_result is _MISSING or zero_is_valid or not types_compatible:
            return zero_is_valid
        for label, default in (
            ("default_on_null", null_default),
            ("default_on_error", parse_error),
        ):
            if (
                default.value_valid
                and default.value is not None
                and Decimal(str(default.value)) == 0
            ):
                collector.add(
                    f"{path}.{label}",
                    f"Column {target_column_name!r} rejects zero but uses zero as {label}.",
                )
        return zero_is_valid

    @staticmethod
    def _validate_binary_default(
        value: str,
        encoding: BinaryEncoding,
        label: str,
    ) -> None:
        """Validate one binary default so sibling defaults can be checked independently."""
        try:
            if encoding is BinaryEncoding.BASE64:
                base64.b64decode(value, validate=True)
            elif encoding is BinaryEncoding.HEX:
                # Spark's ``unhex`` accepts empty and odd-length hexadecimal strings (padding the
                # leading nibble) but rejects whitespace. ``bytes.fromhex`` has the inverse edge
                # behavior, so validate Spark's ASCII grammar directly.
                if _ASCII_HEX_PATTERN.fullmatch(value) is None:
                    raise ValueError("not Spark hexadecimal text")
            else:
                value.encode("utf-8")
        except (ValueError, UnicodeError, binascii.Error) as exc:
            raise CompilationError(f"{label} is not valid {encoding.value} binary text.") from exc

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
        parser_type: ParserType | None,
        target_column_name: str,
        globals_validation: _GlobalsValidation,
        parser_path: str,
        collector: _IssueCollector,
    ) -> tuple[tuple[str, ...], tuple[str, ...], bool, BooleanValuesMode]:
        """Resolve Boolean tokens while suppressing checks with invalid prerequisites."""
        if parser_type is not ParserType.BOOLEAN:
            return (
                DEFAULT_BOOLEAN_TRUE_VALUES,
                DEFAULT_BOOLEAN_FALSE_VALUES,
                DEFAULT_BOOLEAN_CASE_SENSITIVE,
                DEFAULT_BOOLEAN_VALUES_MODE,
            )
        globals_config = globals_validation.config
        mode_result = collector.capture(
            f"{parser_path}.boolean_values_mode",
            lambda: self._enum_value(
                BooleanValuesMode,
                payload.get("boolean_values_mode", DEFAULT_BOOLEAN_VALUES_MODE.value),
                "boolean_values_mode",
            ),
        )
        mode_valid = mode_result is not _MISSING
        mode = mode_result if mode_valid else DEFAULT_BOOLEAN_VALUES_MODE
        supplied_true = "true_values" in payload
        supplied_false = "false_values" in payload
        if "boolean_values_mode" in payload and not (supplied_true or supplied_false):
            collector.add(
                f"{parser_path}.boolean_values_mode",
                f"boolean_values_mode for {target_column_name!r} requires true_values "
                "or false_values.",
            )
        column_true_result = (
            collector.capture(
                f"{parser_path}.true_values",
                lambda: self._string_sequence(
                    payload["true_values"],
                    "true_values",
                    allow_empty_values=False,
                ),
            )
            if supplied_true
            else ()
        )
        column_false_result = (
            collector.capture(
                f"{parser_path}.false_values",
                lambda: self._string_sequence(
                    payload["false_values"],
                    "false_values",
                    allow_empty_values=False,
                ),
            )
            if supplied_false
            else ()
        )
        column_true_valid = column_true_result is not _MISSING
        column_false_valid = column_false_result is not _MISSING
        column_true = column_true_result if column_true_valid else ()
        column_false = column_false_result if column_false_valid else ()
        case_sensitive_result = collector.capture(
            f"{parser_path}.boolean_case_sensitive",
            lambda: self._bool(
                payload,
                "boolean_case_sensitive",
                globals_config.boolean_case_sensitive,
            ),
        )
        case_sensitive_valid = case_sensitive_result is not _MISSING and (
            "boolean_case_sensitive" in payload or globals_validation.boolean_case_sensitive_valid
        )
        column_case_override_valid = (
            "boolean_case_sensitive" in payload and case_sensitive_result is not _MISSING
        )
        case_sensitive = (
            case_sensitive_result
            if case_sensitive_result is not _MISSING
            else globals_config.boolean_case_sensitive
        )

        supplied_overlap_valid = True
        if supplied_true and supplied_false and column_true_valid and column_false_valid:
            supplied_overlap_result = collector.capture(
                parser_path,
                lambda: self._validate_boolean_overlap(
                    column_true,
                    column_false,
                    case_sensitive if case_sensitive_valid else True,
                    f"target column {target_column_name!r}",
                ),
            )
            supplied_overlap_valid = supplied_overlap_result is not _MISSING
        if mode is BooleanValuesMode.EXTEND:
            # Ordered de-duplication keeps serialization and report precedence deterministic.
            true_values = self._deduplicate((*globals_config.true_values, *column_true))
            false_values = self._deduplicate((*globals_config.false_values, *column_false))
            true_values_valid = (
                mode_valid
                and globals_validation.true_values_valid
                and (not supplied_true or column_true_valid)
            )
            false_values_valid = (
                mode_valid
                and globals_validation.false_values_valid
                and (not supplied_false or column_false_valid)
            )
        else:
            true_values = column_true if supplied_true else globals_config.true_values
            false_values = column_false if supplied_false else globals_config.false_values
            true_values_valid = mode_valid and (
                column_true_valid if supplied_true else globals_validation.true_values_valid
            )
            false_values_valid = mode_valid and (
                column_false_valid if supplied_false else globals_validation.false_values_valid
            )
        vocabularies_nonempty = not (
            (true_values_valid and not true_values) or (false_values_valid and not false_values)
        )
        if not vocabularies_nonempty:
            collector.add(
                parser_path,
                "true_values and false_values must be non-empty.",
            )
        should_recheck_inherited_case = (
            mode_valid
            and column_case_override_valid
            and globals_validation.true_values_valid
            and globals_validation.false_values_valid
            and (
                not globals_validation.boolean_case_sensitive_valid
                or case_sensitive != globals_config.boolean_case_sensitive
            )
            and (
                mode is BooleanValuesMode.EXTEND
                or ("boolean_values_mode" not in payload and not (supplied_true or supplied_false))
            )
        )
        if should_recheck_inherited_case:
            inherited_overlap = self._boolean_overlap_values(
                globals_config.true_values,
                globals_config.false_values,
                case_sensitive,
            )
            already_reported = set(globals_validation.boolean_overlap_values)
            if not case_sensitive:
                already_reported.update(
                    {value.lower() for value in already_reported if value.isascii()}
                )
            inherited_overlap.difference_update(already_reported)
            if inherited_overlap:
                collector.add(
                    parser_path,
                    self._boolean_overlap_message(
                        inherited_overlap,
                        f"target column {target_column_name!r}",
                    ),
                )
        if (
            mode_valid
            and mode is BooleanValuesMode.REPLACE
            and (supplied_true or supplied_false)
            and true_values_valid
            and false_values_valid
            and vocabularies_nonempty
            and supplied_overlap_valid
        ):
            collector.capture(
                parser_path,
                lambda: self._validate_boolean_overlap(
                    true_values,
                    false_values,
                    case_sensitive if case_sensitive_valid else True,
                    f"target column {target_column_name!r}",
                ),
            )
        if mode_valid and mode is BooleanValuesMode.EXTEND:
            self._validate_extended_boolean_additions(
                column_true=column_true,
                column_false=column_false,
                column_true_valid=column_true_valid,
                column_false_valid=column_false_valid,
                supplied_true=supplied_true,
                supplied_false=supplied_false,
                globals_validation=globals_validation,
                case_sensitive=case_sensitive,
                case_sensitive_valid=case_sensitive_valid,
                target_column_name=target_column_name,
                parser_path=parser_path,
                collector=collector,
            )
        if not mode_valid and supplied_overlap_valid:
            self._validate_indeterminate_boolean_mode_overlap(
                column_true=column_true,
                column_false=column_false,
                column_true_valid=column_true_valid,
                column_false_valid=column_false_valid,
                supplied_true=supplied_true,
                supplied_false=supplied_false,
                globals_validation=globals_validation,
                case_sensitive=case_sensitive,
                case_sensitive_valid=case_sensitive_valid,
                target_column_name=target_column_name,
                parser_path=parser_path,
                collector=collector,
            )
        return true_values, false_values, case_sensitive, mode

    def _validate_extended_boolean_additions(
        self,
        *,
        column_true: tuple[str, ...],
        column_false: tuple[str, ...],
        column_true_valid: bool,
        column_false_valid: bool,
        supplied_true: bool,
        supplied_false: bool,
        globals_validation: _GlobalsValidation,
        case_sensitive: bool,
        case_sensitive_valid: bool,
        target_column_name: str,
        parser_path: str,
        collector: _IssueCollector,
    ) -> None:
        """Report cross-overlap introduced by EXTEND even when a global sibling is invalid."""
        comparison_is_case_sensitive = case_sensitive if case_sensitive_valid else True
        inherited_true = (
            globals_validation.config.true_values if globals_validation.true_values_valid else ()
        )
        inherited_false = (
            globals_validation.config.false_values if globals_validation.false_values_valid else ()
        )
        new_column_true = self._boolean_additions(
            column_true,
            inherited_true,
            comparison_is_case_sensitive,
        )
        new_column_false = self._boolean_additions(
            column_false,
            inherited_false,
            comparison_is_case_sensitive,
        )
        overlap: set[str] = set()
        if supplied_true and column_true_valid and globals_validation.false_values_valid:
            overlap.update(
                self._boolean_overlap_values(
                    new_column_true,
                    globals_validation.config.false_values,
                    comparison_is_case_sensitive,
                )
            )
        if supplied_false and column_false_valid and globals_validation.true_values_valid:
            overlap.update(
                self._boolean_overlap_values(
                    globals_validation.config.true_values,
                    new_column_false,
                    comparison_is_case_sensitive,
                )
            )
        if supplied_true and supplied_false and column_true_valid and column_false_valid:
            # A direct local overlap was already reported above; keep this diagnostic specific to
            # additional inherited/local interactions so the same finding is not emitted twice.
            overlap.difference_update(
                self._boolean_overlap_values(
                    column_true,
                    column_false,
                    comparison_is_case_sensitive,
                )
            )
        if overlap:
            collector.add(
                parser_path,
                self._boolean_overlap_message(
                    overlap,
                    f"target column {target_column_name!r}",
                ),
            )

    @staticmethod
    def _boolean_additions(
        column_values: tuple[str, ...],
        inherited_values: tuple[str, ...],
        case_sensitive: bool,
    ) -> tuple[str, ...]:
        """Return values that add a new stable token to one side of an inherited vocabulary."""
        exact_inherited = set(inherited_values)
        ascii_inherited = {value.lower() for value in inherited_values if value.isascii()}
        return tuple(
            value
            for value in column_values
            if value not in exact_inherited
            and (case_sensitive or not value.isascii() or value.lower() not in ascii_inherited)
        )

    def _validate_indeterminate_boolean_mode_overlap(
        self,
        *,
        column_true: tuple[str, ...],
        column_false: tuple[str, ...],
        column_true_valid: bool,
        column_false_valid: bool,
        supplied_true: bool,
        supplied_false: bool,
        globals_validation: _GlobalsValidation,
        case_sensitive: bool,
        case_sensitive_valid: bool,
        target_column_name: str,
        parser_path: str,
        collector: _IssueCollector,
    ) -> None:
        """Report overlap that exists under every valid interpretation of an invalid mode.

        Replacement is the smaller of the two possible vocabularies. If its selected true and
        false sides overlap, extension can only retain that overlap, so the issue does not depend
        on which valid mode the author intended.
        """
        if not (supplied_true or supplied_false):
            return
        true_values = column_true if supplied_true else globals_validation.config.true_values
        false_values = column_false if supplied_false else globals_validation.config.false_values
        true_values_valid = (
            column_true_valid if supplied_true else globals_validation.true_values_valid
        )
        false_values_valid = (
            column_false_valid if supplied_false else globals_validation.false_values_valid
        )
        if not true_values_valid or not false_values_valid or not true_values or not false_values:
            return
        collector.capture(
            parser_path,
            lambda: self._validate_boolean_overlap(
                true_values,
                false_values,
                case_sensitive if case_sensitive_valid else True,
                f"target column {target_column_name!r}",
            ),
        )

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
        overlap = YamlParserConfigCompiler._boolean_overlap_values(
            true_values,
            false_values,
            case_sensitive,
        )
        if overlap:
            raise CompilationError(
                YamlParserConfigCompiler._boolean_overlap_message(overlap, label)
            )

    @staticmethod
    def _boolean_overlap_message(overlap: set[str], label: str) -> str:
        """Render one deterministic Boolean vocabulary contradiction."""
        return f"Boolean true_values and false_values overlap for {label}: {sorted(overlap)}."

    @staticmethod
    def _boolean_overlap_values(
        true_values: tuple[str, ...],
        false_values: tuple[str, ...],
        case_sensitive: bool,
    ) -> set[str]:
        """Return overlap labels using the package's stable legacy rendering contract."""
        overlap = set(true_values) & set(false_values)
        if not case_sensitive:
            ascii_true = {item.lower() for item in true_values if item.isascii()}
            ascii_false = {item.lower() for item in false_values if item.isascii()}
            overlap.update(ascii_true & ascii_false)
        return overlap

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
            raise CompilationError(
                f"{label} must contain only valid strings: {_recursion_safe_repr(invalid)}."
            )
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
            raise CompilationError(
                f"{label} keys must be strings: {_recursion_safe_repr(invalid_keys)}."
            )
        for key in value:
            _validate_utf8_string(key, f"{label} key")
        return value

    @staticmethod
    def _reject_keys(payload: Mapping[str, Any], allowed: set[str], label: str) -> None:
        """Fail closed on misspelled or unsupported authoring keys."""
        unsupported = sorted(set(payload) - allowed)
        if unsupported:
            raise CompilationError(f"{label} contains unsupported keys: {unsupported}.")
