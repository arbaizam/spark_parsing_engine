"""Single source of truth for parser authoring defaults.

Compiler logic, metadata discovery, serialization, audits, and documentation all import values
from this module. A default must never be duplicated as an unrelated literal elsewhere; changing
one here should make the effective behavior and its public description change together.
"""

from typing import Final

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
DEFAULT_DATE_FORMATS: Final[tuple[str, ...]] = ("yyyy-MM-dd",)
DEFAULT_TIMESTAMP_FORMATS: Final[tuple[str, ...]] = ("yyyy-MM-dd HH:mm:ss",)
DEFAULT_TIMESTAMP_NTZ_FORMATS: Final[tuple[str, ...]] = DEFAULT_TIMESTAMP_FORMATS
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

# JSON-compatible public view used by documentation and authoring clients. Lists are intentional
# here: callers expect JSON-shaped data, while the immutable runtime models use tuples.
PARSER_DEFAULTS: Final = {
    "globals": {
        "null_markers": list(DEFAULT_NULL_MARKERS),
        "null_marker_case_sensitive": DEFAULT_NULL_MARKER_CASE_SENSITIVE,
        "true_values": list(DEFAULT_BOOLEAN_TRUE_VALUES),
        "false_values": list(DEFAULT_BOOLEAN_FALSE_VALUES),
        "boolean_case_sensitive": DEFAULT_BOOLEAN_CASE_SENSITIVE,
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
    "timestamp_ntz": {"formats": list(DEFAULT_TIMESTAMP_NTZ_FORMATS)},
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
        "true_values": list(DEFAULT_BOOLEAN_TRUE_VALUES),
        "false_values": list(DEFAULT_BOOLEAN_FALSE_VALUES),
        "boolean_case_sensitive": DEFAULT_BOOLEAN_CASE_SENSITIVE,
        "boolean_values_mode": DEFAULT_BOOLEAN_VALUES_MODE.value,
    },
}
