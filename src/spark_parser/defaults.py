"""Single source of truth for parser authoring defaults.

Compiler logic, metadata discovery, serialization, audits, and documentation all import values
from this module. A default must never be duplicated as an unrelated literal elsewhere; changing
one here should make the effective behavior and its public description change together.
"""

from collections.abc import Mapping
from typing import Any, Final, NoReturn

from spark_parser.enums import (
    BinaryEncoding,
    BooleanValuesMode,
    ChildErrorMode,
    ComplexInputFormat,
    NullMarkersMode,
    ParseErrorMode,
)

DEFAULT_NULL_MARKERS: Final[tuple[str, ...]] = ()
DEFAULT_NULL_MARKER_CASE_SENSITIVE: Final = True

# Common normalization and final-value behavior. These defaults favor explicit failures for bad
# non-null input while allowing genuinely absent values to remain null.
DEFAULT_TRIM_WHITESPACE: Final = True
DEFAULT_COLLAPSE_WHITESPACE: Final = True
DEFAULT_EMPTY_IS_NULL: Final = True
DEFAULT_REPLACE_NULL_MARKERS: Final = False
DEFAULT_NULL_MARKERS_MODE: Final = NullMarkersMode.REPLACE
DEFAULT_IS_NULLABLE: Final = True
DEFAULT_ON_PARSE_ERROR: Final = ParseErrorMode.FAIL
DEFAULT_AUDIT: Final = False

# Parser-specific scalar defaults.
DEFAULT_ZERO_IS_VALID: Final = True
DEFAULT_STRING_FORMAT: Final = None
US_MONTH_FIRST_12_HOUR_FORMAT: Final = "MM/dd/yyyy h:mm a"
US_MONTH_FIRST_12_HOUR_SECONDS_FORMAT: Final = "MM/dd/yyyy h:mm:ss a"
ISO_LOCAL_TIMESTAMP_FORMAT: Final = "yyyy-MM-dd'T'HH:mm:ss[.SSSSSS]"
ISO_OFFSET_TIMESTAMP_FORMAT: Final = "yyyy-MM-dd'T'HH:mm:ss[.SSSSSS]XXX"
SQL_LOCAL_TIMESTAMP_FORMAT: Final = "yyyy-MM-dd HH:mm:ss[.SSSSSS]"
# Spark 3.5's default EXCEPTION time-parser policy can throw even through try_to_timestamp when one
# pattern accepts only a prefix. Keep every built-in pattern paired with a full-token shape so
# defaults and runtime guards cannot drift apart.
BUILTIN_DATETIME_FORMAT_SHAPES: Final[dict[str, str]] = {
    "yyyy-MM-dd": r"^\d{4}-\d{2}-\d{2}$",
    ISO_LOCAL_TIMESTAMP_FORMAT: (r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d{1,6})?$"),
    ISO_OFFSET_TIMESTAMP_FORMAT: (
        r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d{1,6})?"
        r"(?:Z|[+-]\d{2}:\d{2})$"
    ),
    SQL_LOCAL_TIMESTAMP_FORMAT: (r"^\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}(?:\.\d{1,6})?$"),
    US_MONTH_FIRST_12_HOUR_FORMAT: (r"^\d{2}/\d{2}/\d{4} \d{1,2}:\d{2} [AaPp][Mm]$"),
    US_MONTH_FIRST_12_HOUR_SECONDS_FORMAT: (r"^\d{2}/\d{2}/\d{4} \d{1,2}:\d{2}:\d{2} [AaPp][Mm]$"),
}
# Date inputs often arrive as an ISO date/local timestamp, a SQL-style local timestamp, or a US
# reporting-system timestamp whose time is irrelevant to the target date. Keep ISO first because it
# is unambiguous. Offset-bearing timestamps are intentionally excluded: converting an instant to a
# date would depend on ``spark.sql.session.timeZone`` and could shift the calendar day. The
# slash-based fallback is deliberately month-first and requires both a time and an AM/PM marker, so
# this default does not pretend to infer ambiguous bare values such as 01/02/2026.
DEFAULT_DATE_FORMATS: Final[tuple[str, ...]] = (
    "yyyy-MM-dd",
    ISO_LOCAL_TIMESTAMP_FORMAT,
    SQL_LOCAL_TIMESTAMP_FORMAT,
    US_MONTH_FIRST_12_HOUR_FORMAT,
    US_MONTH_FIRST_12_HOUR_SECONDS_FORMAT,
)
# Timestamp parsers accept common ISO-8601 shapes as well as the known US export without sacrificing
# the time component. Offset-bearing input belongs only to the ordinary timestamp parser: a
# timestamp_ntz is a local wall-clock value and must never silently discard or reinterpret an offset.
DEFAULT_TIMESTAMP_FORMATS: Final[tuple[str, ...]] = (
    ISO_OFFSET_TIMESTAMP_FORMAT,
    ISO_LOCAL_TIMESTAMP_FORMAT,
    SQL_LOCAL_TIMESTAMP_FORMAT,
    US_MONTH_FIRST_12_HOUR_FORMAT,
    US_MONTH_FIRST_12_HOUR_SECONDS_FORMAT,
)
DEFAULT_TIMESTAMP_NTZ_FORMATS: Final[tuple[str, ...]] = (
    ISO_LOCAL_TIMESTAMP_FORMAT,
    SQL_LOCAL_TIMESTAMP_FORMAT,
    US_MONTH_FIRST_12_HOUR_FORMAT,
    US_MONTH_FIRST_12_HOUR_SECONDS_FORMAT,
)
DEFAULT_BINARY_ENCODING: Final = BinaryEncoding.BASE64
DEFAULT_BOOLEAN_TRUE_VALUES: Final[tuple[str, ...]] = ("true",)
DEFAULT_BOOLEAN_FALSE_VALUES: Final[tuple[str, ...]] = ("false",)
DEFAULT_BOOLEAN_CASE_SENSITIVE: Final = False
DEFAULT_BOOLEAN_VALUES_MODE: Final = BooleanValuesMode.REPLACE

# Container defaults. Child failures fail closed unless an author chooses null/drop explicitly.
DEFAULT_COMPLEX_INPUT_FORMAT: Final = ComplexInputFormat.JSON
DEFAULT_CHILD_ERROR_MODE: Final = ChildErrorMode.FAIL
DEFAULT_DROP_NULL_ELEMENTS: Final = False
DEFAULT_ARRAY_DISTINCT: Final = False
DEFAULT_DROP_NULL_VALUES: Final = False

# Immutable canonical view used by metadata and authoring clients. ``parser.defaults()`` returns a
# detached JSON-shaped copy with lists, while this exported mapping cannot be mutated process-wide
# and made to disagree with the compiler's individual constants above.
_PARSER_DEFAULTS: Final[dict[str, dict[str, Any]]] = {
    "globals": {
        "null_markers": DEFAULT_NULL_MARKERS,
        "null_marker_case_sensitive": DEFAULT_NULL_MARKER_CASE_SENSITIVE,
        "true_values": DEFAULT_BOOLEAN_TRUE_VALUES,
        "false_values": DEFAULT_BOOLEAN_FALSE_VALUES,
        "boolean_case_sensitive": DEFAULT_BOOLEAN_CASE_SENSITIVE,
    },
    "common": {
        "collapse_whitespace": DEFAULT_COLLAPSE_WHITESPACE,
        "trim_whitespace": DEFAULT_TRIM_WHITESPACE,
        "empty_is_null": DEFAULT_EMPTY_IS_NULL,
        "replace_null_markers": DEFAULT_REPLACE_NULL_MARKERS,
        "null_markers": DEFAULT_NULL_MARKERS,
        "null_markers_mode": DEFAULT_NULL_MARKERS_MODE.value,
        "null_marker_case_sensitive": DEFAULT_NULL_MARKER_CASE_SENSITIVE,
        "is_nullable": DEFAULT_IS_NULLABLE,
        "on_parse_error": DEFAULT_ON_PARSE_ERROR.value,
        "audit": DEFAULT_AUDIT,
    },
    "string": {"format": DEFAULT_STRING_FORMAT},
    "numeric": {"zero_is_valid": DEFAULT_ZERO_IS_VALID},
    "date": {"formats": DEFAULT_DATE_FORMATS},
    "timestamp": {"formats": DEFAULT_TIMESTAMP_FORMATS},
    "timestamp_ntz": {"formats": DEFAULT_TIMESTAMP_NTZ_FORMATS},
    "binary": {"encoding": DEFAULT_BINARY_ENCODING.value},
    "array": {
        "collapse_whitespace": False,
        "input_format": DEFAULT_COMPLEX_INPUT_FORMAT.value,
        "on_element_error": DEFAULT_CHILD_ERROR_MODE.value,
        "drop_null_elements": DEFAULT_DROP_NULL_ELEMENTS,
        "distinct": DEFAULT_ARRAY_DISTINCT,
    },
    "struct": {
        "collapse_whitespace": False,
        "input_format": DEFAULT_COMPLEX_INPUT_FORMAT.value,
    },
    "map": {
        "collapse_whitespace": False,
        "input_format": DEFAULT_COMPLEX_INPUT_FORMAT.value,
        "on_value_error": DEFAULT_CHILD_ERROR_MODE.value,
        "drop_null_values": DEFAULT_DROP_NULL_VALUES,
    },
    "boolean": {
        "true_values": DEFAULT_BOOLEAN_TRUE_VALUES,
        "false_values": DEFAULT_BOOLEAN_FALSE_VALUES,
        "boolean_case_sensitive": DEFAULT_BOOLEAN_CASE_SENSITIVE,
        "boolean_values_mode": DEFAULT_BOOLEAN_VALUES_MODE.value,
    },
}


class _ImmutableDefaults(dict[str, Any]):
    """JSON-serializable dictionary that rejects process-wide default mutation."""

    @staticmethod
    def _reject_mutation(*_args: Any, **_kwargs: Any) -> NoReturn:
        raise TypeError("PARSER_DEFAULTS is immutable; call parser.defaults() for a mutable copy.")

    __setitem__ = _reject_mutation
    __delitem__ = _reject_mutation
    clear = _reject_mutation
    pop = _reject_mutation
    popitem = _reject_mutation
    setdefault = _reject_mutation
    update = _reject_mutation
    __ior__ = _reject_mutation


class _ImmutableList(list[Any]):
    """List-compatible default value that cannot mutate shared package state."""

    @staticmethod
    def _reject_mutation(*_args: Any, **_kwargs: Any) -> NoReturn:
        raise TypeError("PARSER_DEFAULTS is immutable; call parser.defaults() for a mutable copy.")

    __setitem__ = _reject_mutation
    __delitem__ = _reject_mutation
    __iadd__ = _reject_mutation
    __imul__ = _reject_mutation
    append = _reject_mutation
    clear = _reject_mutation
    extend = _reject_mutation
    insert = _reject_mutation
    pop = _reject_mutation
    remove = _reject_mutation
    reverse = _reject_mutation
    sort = _reject_mutation


PARSER_DEFAULTS: Final[Mapping[str, Mapping[str, Any]]] = _ImmutableDefaults(
    {
        section: _ImmutableDefaults(
            {
                key: _ImmutableList(value) if isinstance(value, tuple) else value
                for key, value in values.items()
            }
        )
        for section, values in _PARSER_DEFAULTS.items()
    }
)


def parser_defaults() -> dict[str, dict[str, Any]]:
    """Return a detached JSON-compatible copy of every effective authoring default."""
    return {
        section: {
            key: list(value) if isinstance(value, tuple) else value for key, value in values.items()
        }
        for section, values in _PARSER_DEFAULTS.items()
    }
