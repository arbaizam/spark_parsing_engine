"""Single source of truth for parser authoring defaults."""

from typing import Final

from spark_parser.enums import NullMarkersMode, ParseErrorMode

DEFAULT_NULL_MARKERS: Final[tuple[str, ...]] = ()
DEFAULT_NULL_MARKER_CASE_SENSITIVE: Final = True

DEFAULT_TRIM_WHITESPACE: Final = True
DEFAULT_COLLAPSE_WHITESPACE: Final = True
DEFAULT_EMPTY_IS_NULL: Final = True
DEFAULT_REPLACE_NULL_MARKERS: Final = False
DEFAULT_NULL_MARKERS_MODE: Final = NullMarkersMode.REPLACE
DEFAULT_IS_NULLABLE: Final = True
DEFAULT_ON_PARSE_ERROR: Final = ParseErrorMode.FAIL
DEFAULT_AUDIT: Final = False

DEFAULT_ZERO_IS_VALID: Final = True
DEFAULT_STRING_FORMAT: Final = None
DEFAULT_DATE_FORMATS: Final[tuple[str, ...]] = ("yyyy-MM-dd",)
DEFAULT_TIMESTAMP_FORMATS: Final[tuple[str, ...]] = ("yyyy-MM-dd HH:mm:ss",)
DEFAULT_BOOLEAN_TRUE_VALUES: Final[tuple[str, ...]] = ("true",)
DEFAULT_BOOLEAN_FALSE_VALUES: Final[tuple[str, ...]] = ("false",)
DEFAULT_BOOLEAN_CASE_SENSITIVE: Final = False

# JSON-compatible public view used by documentation and authoring clients.
PARSER_DEFAULTS: Final = {
    "globals": {
        "null_markers": list(DEFAULT_NULL_MARKERS),
        "null_marker_case_sensitive": DEFAULT_NULL_MARKER_CASE_SENSITIVE,
    },
    "common": {
        "collapse_whitespace": DEFAULT_COLLAPSE_WHITESPACE,
        "trim_whitespace": DEFAULT_TRIM_WHITESPACE,
        "empty_is_null": DEFAULT_EMPTY_IS_NULL,
        "replace_null_markers": DEFAULT_REPLACE_NULL_MARKERS,
        "null_markers": list(DEFAULT_NULL_MARKERS),
        "null_markers_mode": DEFAULT_NULL_MARKERS_MODE.value,
        "null_marker_case_sensitive": DEFAULT_NULL_MARKER_CASE_SENSITIVE,
        "is_nullable": DEFAULT_IS_NULLABLE,
        "on_parse_error": DEFAULT_ON_PARSE_ERROR.value,
        "audit": DEFAULT_AUDIT,
    },
    "string": {"format": DEFAULT_STRING_FORMAT},
    "numeric": {"zero_is_valid": DEFAULT_ZERO_IS_VALID},
    "date": {"formats": list(DEFAULT_DATE_FORMATS)},
    "timestamp": {"formats": list(DEFAULT_TIMESTAMP_FORMATS)},
    "boolean": {
        "true_values": list(DEFAULT_BOOLEAN_TRUE_VALUES),
        "false_values": list(DEFAULT_BOOLEAN_FALSE_VALUES),
        "boolean_case_sensitive": DEFAULT_BOOLEAN_CASE_SENSITIVE,
    },
}
