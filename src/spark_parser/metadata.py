"""Machine-readable authoring guidance exposed by the public service API.

This metadata is built from the same enum vocabulary and defaults used by compilation. Notebook
help, documentation generators, and configuration UIs can therefore explain actual behavior
without maintaining an independent, drift-prone argument catalog.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any

from spark_parser.defaults import parser_defaults
from spark_parser.enums import (
    NUMERIC_PARSER_TYPES,
    BinaryEncoding,
    ChildErrorMode,
    ComplexInputFormat,
    ParseErrorMode,
    ParserType,
    StringFormat,
)

_PARSER_DEFAULTS = parser_defaults()


def _argument(
    name: str,
    *,
    required: bool = False,
    default: Any = None,
    default_kind: str = "literal",
    allowed_values: list[str] | None = None,
    description: str,
    condition: str | None = None,
) -> dict[str, Any]:
    """Create one detached argument-description record with a consistent public shape."""
    return {
        "name": name,
        "required": required,
        "condition": condition,
        "default": deepcopy(default),
        "default_kind": default_kind,
        "allowed_values": allowed_values,
        "description": description,
    }


# Arguments shared by every parser node. Parser-specific lists below are appended to a deep copy so
# callers can safely modify returned metadata without mutating these module constants.
_COMMON_ARGUMENTS = [
    _argument(
        "type",
        condition="required in parser mapping form; scalar parser form supplies it directly",
        description="Parser implementation; must agree with expected_data_type.",
    ),
    _argument(
        "collapse_whitespace",
        default=_PARSER_DEFAULTS["common"]["collapse_whitespace"],
        description="Collapse every run of whitespace, including internal whitespace, to one space.",
    ),
    _argument(
        "trim_whitespace",
        default=_PARSER_DEFAULTS["common"]["trim_whitespace"],
        description=(
            "Remove leading and trailing spaces, tabs, line breaks, and non-breaking spaces "
            "after collapse_whitespace."
        ),
    ),
    _argument(
        "empty_is_null",
        default=_PARSER_DEFAULTS["common"]["empty_is_null"],
        description="Convert an empty normalized string to null.",
    ),
    _argument(
        "replace_null_markers",
        default=_PARSER_DEFAULTS["common"]["replace_null_markers"],
        description="Convert matching null-marker strings to null; markers are inert when false.",
    ),
    _argument(
        "null_markers",
        default=_PARSER_DEFAULTS["common"]["null_markers"],
        default_kind="inherited_global",
        description="Column null tokens; inherited from globals when omitted.",
    ),
    _argument(
        "null_markers_mode",
        default=_PARSER_DEFAULTS["common"]["null_markers_mode"],
        allowed_values=["replace", "extend"],
        description="Replace or extend global null markers when column null_markers are supplied.",
    ),
    _argument(
        "null_marker_case_sensitive",
        default=_PARSER_DEFAULTS["common"]["null_marker_case_sensitive"],
        default_kind="inherited_global",
        description="Use exact-case null matching when true; compare lowercase values when false.",
    ),
    _argument(
        "is_nullable",
        default=_PARSER_DEFAULTS["common"]["is_nullable"],
        description="Allow the final target value to remain null.",
    ),
    _argument(
        "default_on_null",
        condition="required when is_nullable is false",
        description="Typed value assigned after all parsing and zero handling when output is null.",
    ),
    _argument(
        "on_parse_error",
        default=_PARSER_DEFAULTS["common"]["on_parse_error"],
        allowed_values=["fail", "null", "default"],
        description=(
            "Raise at Spark action time, return null, or assign default_on_error. String parsers "
            "also allow preserve to return the exact raw token."
        ),
    ),
    _argument(
        "default_on_error",
        condition="required when on_parse_error is default",
        description="Typed value assigned only when a non-null normalized value cannot parse.",
    ),
    _argument(
        "audit",
        default=_PARSER_DEFAULTS["common"]["audit"],
        description="Include row-level details for this column in the audit struct array.",
    ),
]


def _zero_is_valid_argument() -> dict[str, Any]:
    """Return a fresh shared numeric-argument description."""
    return _argument(
        "zero_is_valid",
        default=_PARSER_DEFAULTS["numeric"]["zero_is_valid"],
        description="Keep zero when true; convert zero to null before default_on_null when false.",
    )


# Parser-specific authoring arguments. Keeping this table declarative makes omissions visible in
# code review whenever a new ParserType is introduced.
_SPECIFIC_ARGUMENTS: dict[ParserType, list[dict[str, Any]]] = {
    **{parser_type: [_zero_is_valid_argument()] for parser_type in NUMERIC_PARSER_TYPES},
    ParserType.STRING: [
        _argument(
            "format",
            default=_PARSER_DEFAULTS["string"]["format"],
            allowed_values=["null", "none", *(member.value for member in StringFormat)],
            description="Optional deterministic string formatting profile.",
        )
    ],
    ParserType.BINARY: [
        _argument(
            "encoding",
            default=_PARSER_DEFAULTS["binary"]["encoding"],
            allowed_values=[member.value for member in BinaryEncoding],
            description="Decode the normalized bronze string as base64, hexadecimal, or UTF-8 bytes.",
        )
    ],
    ParserType.BOOLEAN: [
        _argument(
            "true_values",
            default=_PARSER_DEFAULTS["boolean"]["true_values"],
            default_kind="inherited_global",
            description="Non-empty tokens mapped to true.",
        ),
        _argument(
            "false_values",
            default=_PARSER_DEFAULTS["boolean"]["false_values"],
            default_kind="inherited_global",
            description="Non-empty tokens mapped to false; may not overlap true_values.",
        ),
        _argument(
            "boolean_values_mode",
            default=_PARSER_DEFAULTS["boolean"]["boolean_values_mode"],
            allowed_values=["replace", "extend"],
            description="Replace or extend global Boolean tokens when column tokens are supplied.",
        ),
        _argument(
            "boolean_case_sensitive",
            default=_PARSER_DEFAULTS["boolean"]["boolean_case_sensitive"],
            default_kind="inherited_global",
            description="Use exact-case Boolean-token matching when true.",
        ),
    ],
    ParserType.DATE: [
        _argument(
            "formats",
            default=_PARSER_DEFAULTS["date"]["formats"],
            description=(
                "Non-empty Spark datetime patterns tried in list order; first success preserves "
                "the authored calendar fields without session-timezone conversion."
            ),
        )
    ],
    ParserType.TIMESTAMP: [
        _argument(
            "formats",
            default=_PARSER_DEFAULTS["timestamp"]["formats"],
            description="Non-empty Spark datetime patterns tried in list order; first success wins.",
        )
    ],
    ParserType.TIMESTAMP_NTZ: [
        _argument(
            "formats",
            default=_PARSER_DEFAULTS["timestamp_ntz"]["formats"],
            description="Non-empty Spark datetime patterns tried in list order without timezone conversion.",
        )
    ],
    ParserType.ARRAY: [
        _argument(
            "input_format",
            default=_PARSER_DEFAULTS["array"]["input_format"],
            allowed_values=[member.value for member in ComplexInputFormat],
            description="Parse a JSON array or split a scalar array using a literal delimiter.",
        ),
        _argument(
            "delimiter",
            condition="required when input_format is delimited",
            description="Literal separator for a delimited scalar array.",
        ),
        _argument(
            "element_parser",
            required=True,
            description="Recursive parser matching the element type in expected_data_type.",
        ),
        _argument(
            "on_element_error",
            default=_PARSER_DEFAULTS["array"]["on_element_error"],
            allowed_values=[member.value for member in ChildErrorMode],
            description=(
                "Handle a direct element-parser failure by failing the row, routing null through "
                "element final-null handling, dropping the element, or preserving its raw token "
                "when the element type is string. Successfully decoded complex elements retain "
                "their descendants' own error policies."
            ),
        ),
        _argument(
            "drop_null_elements",
            default=_PARSER_DEFAULTS["array"]["drop_null_elements"],
            description="Remove source nulls and values resolved to null after parsing.",
        ),
        _argument(
            "distinct",
            default=_PARSER_DEFAULTS["array"]["distinct"],
            description="Remove duplicate parsed elements while preserving first occurrence order.",
        ),
    ],
    ParserType.STRUCT: [
        _argument(
            "input_format",
            default=_PARSER_DEFAULTS["struct"]["input_format"],
            allowed_values=[ComplexInputFormat.JSON.value],
            description="Struct values are parsed from JSON objects.",
        ),
        _argument(
            "fields",
            required=True,
            description=(
                "Complete ordered source-to-target field mapping with recursive parsers; source "
                "and target names are preserved verbatim."
            ),
        ),
    ],
    ParserType.MAP: [
        _argument(
            "input_format",
            default=_PARSER_DEFAULTS["map"]["input_format"],
            allowed_values=[ComplexInputFormat.JSON.value],
            description="Map values are parsed from JSON objects with string keys.",
        ),
        _argument(
            "value_parser",
            required=True,
            description="Recursive parser matching the map value type in expected_data_type.",
        ),
        _argument(
            "on_value_error",
            default=_PARSER_DEFAULTS["map"]["on_value_error"],
            allowed_values=[member.value for member in ChildErrorMode],
            description=(
                "Handle a direct value-parser failure by failing the row, routing null through "
                "value final-null handling, dropping the entry, or preserving its raw value when "
                "the map value type is string. Successfully decoded complex values retain their "
                "descendants' own error policies."
            ),
        ),
        _argument(
            "drop_null_values",
            default=_PARSER_DEFAULTS["map"]["drop_null_values"],
            description="Remove entries whose parsed value is null.",
        ),
    ],
}


# Human-facing summaries and guidance remain data rather than branching prose in service.py.
_SUMMARIES = {
    ParserType.STRING: "Normalize a string and optionally apply a deterministic display profile.",
    ParserType.BYTE: "Parse a bronze string as an 8-bit Spark byte.",
    ParserType.SHORT: "Parse a bronze string as a 16-bit Spark short.",
    ParserType.INTEGER: "Parse a bronze string as a 32-bit Spark integer.",
    ParserType.LONG: "Parse a bronze string as a 64-bit Spark long.",
    ParserType.FLOAT: "Parse a bronze string as a single-precision Spark float.",
    ParserType.DECIMAL: "Parse a bronze string into the configured decimal(p,s).",
    ParserType.DOUBLE: "Parse a bronze string as a Spark double.",
    ParserType.BINARY: "Decode a bronze string into Spark binary bytes.",
    ParserType.BOOLEAN: "Map configured normalized tokens to true or false.",
    ParserType.DATE: "Cascade through configured Spark datetime patterns and return a date.",
    ParserType.TIMESTAMP: "Cascade through configured Spark datetime patterns and return a timestamp.",
    ParserType.TIMESTAMP_NTZ: "Parse a wall-clock timestamp without timezone interpretation.",
    ParserType.ARRAY: "Parse a JSON or delimited array and recursively parse every element.",
    ParserType.STRUCT: "Parse a JSON object into configured, recursively parsed target fields.",
    ParserType.MAP: "Parse a JSON object into a string-keyed map with recursively parsed values.",
}


_SPECIFIC_BEHAVIORS = {
    ParserType.STRING: [
        "format null preserves the whitespace-normalized value.",
        "on_parse_error preserve returns the exact pre-normalization source token when formatting fails.",
        "title lowercases and capitalizes words while retaining normalized spaces.",
        "pascal removes spaces after init-capitalization; it is intended for identifiers, not names.",
        "address_us_v1 uses contextual USPS-style suffixes/directionals and smart-cases Mc, apostrophe, and hyphen names.",
        "county smart-cases the name and ensures exactly one trailing 'County'.",
        "state_us maps one or more comma-separated US state and territory names, postal codes, conventional state abbreviations, and Washington, DC to uppercase two-letter codes.",
        "zip parses one or more comma-separated ZIP5 or ZIP+4 values, retains string output, and pads short numeric components with leading zeroes.",
    ],
    ParserType.BOOLEAN: [
        "Matching occurs after whitespace normalization.",
        "Exact and ASCII case-insensitive true/false overlap is rejected during compilation; non-ASCII case-insensitive overlap is validated by Spark with the runtime's own Unicode tables.",
        "Quote YAML tokens such as 'true', 'false', 'yes', 'no', 'on', and 'off' so they remain strings.",
    ],
    ParserType.DATE: [
        "Formats cascade in order; format inference is not performed.",
        "The defaults accept an ISO date/local timestamp, a SQL-style local timestamp, or a US month-first 12-hour timestamp, with or without seconds, and return only the date.",
        "An explicitly configured offset-bearing format validates the offset but preserves the calendar date written in the source instead of projecting the instant through the Spark session timezone.",
    ],
    ParserType.TIMESTAMP: [
        "Formats cascade in order; format inference is not performed.",
        "The defaults accept local or offset-bearing ISO timestamps, optional microseconds, and US month-first 12-hour timestamps with or without seconds.",
    ],
    ParserType.TIMESTAMP_NTZ: [
        "Formats cascade in order without applying the Spark session timezone.",
        "The defaults accept timezone-free ISO timestamps, optional microseconds, and US month-first 12-hour timestamps with or without seconds.",
    ],
    ParserType.ARRAY: [
        "JSON arrays support recursive element types; delimited arrays support scalar elements.",
        "Element failures are reported with zero-based paths such as $[2].",
        "Nested defaults and invalidated zeros are reported in dedicated top-level audit path arrays.",
    ],
    ParserType.STRUCT: [
        "Every target struct field must have exactly one source field mapping and recursive parser.",
        "Unknown JSON fields are ignored; configured missing fields become null/default.",
        "Nested defaults and invalidated zeros are reported in dedicated top-level audit path arrays.",
    ],
    ParserType.MAP: [
        "JSON map keys are strings and values may use any recursively supported datatype.",
        "Value failures are reported with paths such as $['balance'].",
        "Apostrophes in diagnostic map keys are backslash-escaped so one key remains one path segment.",
    ],
}


_SPECIFIC_GOTCHAS = {
    ParserType.STRING: [
        "address_us_v1 is deterministic display normalization, not postal validation or deliverability verification.",
        "title uses Spark initcap semantics; it does not apply address/name exceptions such as McLean.",
        "county is for jurisdictions named County; it does not infer Parish, Borough, or Census Area.",
        "state_us includes the 50 states, Washington DC, AS, GU, MP, PR, and VI; other unknown non-null values are parse errors.",
        "state_us and zip treat a comma as a property-value separator and fail the whole value when any component is invalid.",
        "zip rejects compact six-to-eight-digit values instead of guessing a ZIP+4 split.",
    ],
    ParserType.DECIMAL: [
        "expected_data_type must include precision and scale; Spark precision is limited to 38.",
        "Spark rounds source values with excess scale to the configured decimal scale.",
        "String defaults use an ASCII decimal/scientific grammar; underscores and surrounding whitespace are rejected.",
    ],
    ParserType.DOUBLE: ["Use decimal(p,s) when exact base-10 representation matters."],
    ParserType.FLOAT: ["Use double or decimal(p,s) when additional precision is required."],
    ParserType.BINARY: [
        "parsed_value audit output is canonical base64 regardless of input encoding."
    ],
    ParserType.BOOLEAN: ["Unknown non-null tokens are parse errors, not false."],
    ParserType.DATE: [
        "Spark datetime patterns are not Python strptime patterns.",
        "The built-in slash-date fallback is explicitly MM/dd/yyyy, not dd/MM/yyyy.",
        "Offset-bearing timestamps are not date defaults; configure their format explicitly when discarding time and offset semantics is intentional.",
        "Formats outside the built-in guarded pattern set require spark.sql.legacy.timeParserPolicy=CORRECTED when binding a DataFrame.",
    ],
    ParserType.TIMESTAMP: [
        "Timestamp interpretation follows the active Spark SQL session timezone.",
        "The built-in slash-date fallback is explicitly MM/dd/yyyy, not dd/MM/yyyy.",
        "Formats outside the built-in guarded pattern set require spark.sql.legacy.timeParserPolicy=CORRECTED when binding a DataFrame.",
    ],
    ParserType.TIMESTAMP_NTZ: [
        "Use timestamp when the value represents an absolute instant rather than local wall-clock time.",
        "Offset-bearing inputs and typed defaults are rejected instead of silently discarding timezone information.",
        "The built-in slash-date fallback is explicitly MM/dd/yyyy, not dd/MM/yyyy.",
        "Formats outside the built-in guarded pattern set require spark.sql.legacy.timeParserPolicy=CORRECTED when binding a DataFrame.",
    ],
    ParserType.ARRAY: [
        "Delimited input treats the delimiter literally and does not implement CSV quoting.",
        "Nested child audit is consolidated into the top-level column audit entry.",
    ],
    ParserType.STRUCT: [
        "Duplicate JSON object keys, including unselected fields, make the whole container a parse "
        "error handled by on_parse_error or the parent child-error policy."
    ],
    ParserType.MAP: [
        "Duplicate JSON object keys make the whole container a parse error handled by "
        "on_parse_error or the parent child-error policy."
    ],
}


def parser_description(parser_type: ParserType) -> dict[str, Any]:
    """Return a fresh machine-readable description for one parser type.

    Complex containers override the common whitespace-collapse default because rewriting raw JSON
    could alter quoted child values. Recursive leaf parsers still advertise their own defaults.
    """
    expected = {
        ParserType.DECIMAL: ["decimal(p,s)"],
        ParserType.ARRAY: ["array<T>"],
        ParserType.STRUCT: ["struct<field:T,...>"],
        ParserType.MAP: ["map<string,T>"],
    }.get(parser_type, [parser_type.value])
    is_complex = parser_type in {ParserType.ARRAY, ParserType.STRUCT, ParserType.MAP}
    normalization_behavior = (
        "Outer complex input is edge-trimmed and collapse_whitespace resolves to false because "
        "collapsing inside JSON would rewrite quoted values; recursive leaf parsers perform "
        "their own normalization."
        if is_complex
        else "Whitespace collapse, trim, and empty-to-null run before parser-specific conversion."
    )
    # Deep copy is part of the public contract: consumers may annotate or reorder metadata locally
    # without changing later calls.
    arguments = deepcopy([*_COMMON_ARGUMENTS, *_SPECIFIC_ARGUMENTS[parser_type]])
    if parser_type is ParserType.STRING:
        # Preserve is intentionally absent from non-string parser metadata because the compiler
        # rejects raw fallback whenever the target position cannot legally contain a string.
        for argument in arguments:
            if argument["name"] == "on_parse_error":
                argument["allowed_values"] = [member.value for member in ParseErrorMode]
                break
    if is_complex:
        for argument in arguments:
            if argument["name"] == "collapse_whitespace":
                argument["default"] = _PARSER_DEFAULTS[parser_type.value]["collapse_whitespace"]
                argument["description"] = (
                    "Always resolves to false for the outer complex container; recursive leaf "
                    "parsers control their own whitespace collapse."
                )
                break
    return {
        "parser_type": parser_type.value,
        "expected_data_types": expected,
        "summary": _SUMMARIES[parser_type],
        "arguments": arguments,
        "key_behaviors": [
            normalization_behavior,
            "Null markers match only when replace_null_markers is true.",
            "Parse errors are resolved before zero and final-null handling.",
            "fail mode raises only when Spark materializes the failing target expression; projection pruning can skip it.",
            "All execution uses native Spark expressions; no Python UDF is used.",
            *(
                ["Only the exact lowercase JSON literal null is treated as a JSON null token."]
                if is_complex
                else []
            ),
            *_SPECIFIC_BEHAVIORS.get(parser_type, []),
        ],
        "gotchas": deepcopy(_SPECIFIC_GOTCHAS.get(parser_type, [])),
    }


def config_description() -> dict[str, Any]:
    """Return a fresh description of top-level, global, and column authoring contracts."""
    return {
        "top_level_arguments": [
            _argument("parser_config_id", required=True, description="Stable configuration ID."),
            _argument("parser_config_name", required=True, description="Human-readable name."),
            _argument("version", required=True, description="Non-empty version string."),
            _argument("description", description="Optional purpose and scope."),
            _argument("owner", description="Optional accountable owner."),
            _argument("owner_department", description="Optional accountable department."),
            _argument(
                "globals", default={}, description="Inherited null and Boolean vocabularies."
            ),
            _argument("columns", required=True, description="Non-empty ordered column mappings."),
        ],
        "global_arguments": [
            _argument("null_markers", default=[], description="Default null-token list."),
            _argument(
                "null_marker_case_sensitive",
                default=_PARSER_DEFAULTS["globals"]["null_marker_case_sensitive"],
                description="Default exact-case null matching behavior.",
            ),
            _argument(
                "true_values",
                default=_PARSER_DEFAULTS["globals"]["true_values"],
                description="Global true-token list.",
            ),
            _argument(
                "false_values",
                default=_PARSER_DEFAULTS["globals"]["false_values"],
                description="Global false-token list.",
            ),
            _argument(
                "boolean_case_sensitive",
                default=_PARSER_DEFAULTS["globals"]["boolean_case_sensitive"],
                description="Whether global Boolean-token matching requires exact case.",
            ),
        ],
        "column_arguments": [
            _argument(
                "source_column_name",
                required=True,
                description=(
                    "Top-level bronze source name preserved verbatim and resolved with Spark's "
                    "active identifier resolver; missing input fails binding unless the caller "
                    "explicitly selects on_missing_source='warn'."
                ),
            ),
            _argument(
                "target_column_name",
                required=True,
                description=(
                    "Non-blank output name preserved verbatim in parsed_df; exact duplicates fail "
                    "compilation and resolver-sensitive collisions fail DataFrame binding."
                ),
            ),
            _argument(
                "expected_data_type",
                required=True,
                description=(
                    "Exact scalar or recursive Spark DDL target type; validated against the "
                    "complete parser tree."
                ),
            ),
            _argument(
                "parser",
                required=True,
                description="Scalar or recursive complex parser type/options mapping.",
            ),
        ],
    }
