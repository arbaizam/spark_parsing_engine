"""Discoverable authoring metadata used by the public service API."""

from __future__ import annotations

from copy import deepcopy
from typing import Any

from spark_parser.defaults import PARSER_DEFAULTS
from spark_parser.enums import ParserType, StringFormat


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
    return {
        "name": name,
        "required": required,
        "condition": condition,
        "default": deepcopy(default),
        "default_kind": default_kind,
        "allowed_values": allowed_values,
        "description": description,
    }


_COMMON_ARGUMENTS = [
    _argument(
        "type",
        condition="required in parser mapping form; scalar parser form supplies it directly",
        description="Parser implementation; must agree with expected_data_type.",
    ),
    _argument(
        "collapse_whitespace",
        default=PARSER_DEFAULTS["common"]["collapse_whitespace"],
        description="Collapse every run of whitespace, including internal whitespace, to one space.",
    ),
    _argument(
        "trim_whitespace",
        default=PARSER_DEFAULTS["common"]["trim_whitespace"],
        description=(
            "Remove leading and trailing spaces, tabs, line breaks, and non-breaking spaces "
            "after collapse_whitespace."
        ),
    ),
    _argument(
        "empty_is_null",
        default=PARSER_DEFAULTS["common"]["empty_is_null"],
        description="Convert an empty normalized string to null.",
    ),
    _argument(
        "replace_null_markers",
        default=PARSER_DEFAULTS["common"]["replace_null_markers"],
        description="Convert matching null-marker strings to null; markers are inert when false.",
    ),
    _argument(
        "null_markers",
        default=PARSER_DEFAULTS["common"]["null_markers"],
        default_kind="inherited_global",
        description="Column null tokens; inherited from globals when omitted.",
    ),
    _argument(
        "null_markers_mode",
        default=PARSER_DEFAULTS["common"]["null_markers_mode"],
        allowed_values=["replace", "extend"],
        description="Replace or extend global null markers when column null_markers are supplied.",
    ),
    _argument(
        "null_marker_case_sensitive",
        default=PARSER_DEFAULTS["common"]["null_marker_case_sensitive"],
        default_kind="inherited_global",
        description="Use exact-case null matching when true; compare lowercase values when false.",
    ),
    _argument(
        "is_nullable",
        default=PARSER_DEFAULTS["common"]["is_nullable"],
        description="Allow the final silver value to remain null.",
    ),
    _argument(
        "default_on_null",
        condition="required when is_nullable is false",
        description="Typed value assigned after all parsing and zero handling when output is null.",
    ),
    _argument(
        "on_parse_error",
        default=PARSER_DEFAULTS["common"]["on_parse_error"],
        allowed_values=["fail", "null", "default"],
        description="Raise at Spark action time, return null, or assign default_on_error.",
    ),
    _argument(
        "default_on_error",
        condition="required when on_parse_error is default",
        description="Typed value assigned only when a non-null normalized value cannot parse.",
    ),
    _argument(
        "audit",
        default=PARSER_DEFAULTS["common"]["audit"],
        description="Include row-level details for this column in the audit struct array.",
    ),
]


_SPECIFIC_ARGUMENTS = {
    ParserType.STRING: [
        _argument(
            "format",
            default=PARSER_DEFAULTS["string"]["format"],
            allowed_values=["null", "none", *(member.value for member in StringFormat)],
            description="Optional deterministic string formatting profile.",
        )
    ],
    ParserType.INTEGER: [
        _argument(
            "zero_is_valid",
            default=PARSER_DEFAULTS["numeric"]["zero_is_valid"],
            description="Keep zero when true; convert zero to null before default_on_null when false.",
        )
    ],
    ParserType.LONG: [
        _argument(
            "zero_is_valid",
            default=PARSER_DEFAULTS["numeric"]["zero_is_valid"],
            description="Keep zero when true; convert zero to null before default_on_null when false.",
        )
    ],
    ParserType.DECIMAL: [
        _argument(
            "zero_is_valid",
            default=PARSER_DEFAULTS["numeric"]["zero_is_valid"],
            description="Keep zero when true; convert zero to null before default_on_null when false.",
        )
    ],
    ParserType.DOUBLE: [
        _argument(
            "zero_is_valid",
            default=PARSER_DEFAULTS["numeric"]["zero_is_valid"],
            description="Keep zero when true; convert zero to null before default_on_null when false.",
        )
    ],
    ParserType.BOOLEAN: [
        _argument(
            "true_values",
            default=PARSER_DEFAULTS["boolean"]["true_values"],
            default_kind="inherited_global",
            description="Non-empty tokens mapped to true.",
        ),
        _argument(
            "false_values",
            default=PARSER_DEFAULTS["boolean"]["false_values"],
            default_kind="inherited_global",
            description="Non-empty tokens mapped to false; may not overlap true_values.",
        ),
        _argument(
            "boolean_values_mode",
            default=PARSER_DEFAULTS["boolean"]["boolean_values_mode"],
            allowed_values=["replace", "extend"],
            description="Replace or extend global Boolean tokens when column tokens are supplied.",
        ),
        _argument(
            "boolean_case_sensitive",
            default=PARSER_DEFAULTS["boolean"]["boolean_case_sensitive"],
            default_kind="inherited_global",
            description="Use exact-case Boolean-token matching when true.",
        ),
    ],
    ParserType.DATE: [
        _argument(
            "formats",
            default=PARSER_DEFAULTS["date"]["formats"],
            description="Non-empty Spark datetime patterns tried in list order; first success wins.",
        )
    ],
    ParserType.TIMESTAMP: [
        _argument(
            "formats",
            default=PARSER_DEFAULTS["timestamp"]["formats"],
            description="Non-empty Spark datetime patterns tried in list order; first success wins.",
        )
    ],
}


_SUMMARIES = {
    ParserType.STRING: "Normalize a string and optionally apply a deterministic display profile.",
    ParserType.INTEGER: "Parse a bronze string as a 32-bit Spark integer.",
    ParserType.LONG: "Parse a bronze string as a 64-bit Spark long.",
    ParserType.DECIMAL: "Parse a bronze string into the configured decimal(p,s).",
    ParserType.DOUBLE: "Parse a bronze string as a Spark double.",
    ParserType.BOOLEAN: "Map configured normalized tokens to true or false.",
    ParserType.DATE: "Cascade through configured Spark datetime patterns and return a date.",
    ParserType.TIMESTAMP: "Cascade through configured Spark datetime patterns and return a timestamp.",
}


_SPECIFIC_BEHAVIORS = {
    ParserType.STRING: [
        "format null preserves the whitespace-normalized value.",
        "pascal removes spaces after init-capitalization; it is intended for identifiers, not names.",
        "address_us_v1 uses contextual USPS-style suffixes/directionals and smart-cases Mc, apostrophe, and hyphen names.",
        "county smart-cases the name and ensures exactly one trailing 'County'.",
        "zip returns ZIP5 or ZIP+4 as a string and pads short numeric components with leading zeroes.",
    ],
    ParserType.BOOLEAN: [
        "Matching occurs after whitespace normalization.",
        "Case-insensitive vocabularies are validated for true/false overlap after lowercasing, matching Spark runtime behavior.",
        "Quote YAML tokens such as 'true', 'false', 'yes', 'no', 'on', and 'off' so they remain strings.",
    ],
    ParserType.DATE: ["Formats cascade in order; format inference is not performed."],
    ParserType.TIMESTAMP: ["Formats cascade in order; format inference is not performed."],
}


_SPECIFIC_GOTCHAS = {
    ParserType.STRING: [
        "address_us_v1 is deterministic display normalization, not postal validation or deliverability verification.",
        "county is for jurisdictions named County; it does not infer Parish, Borough, or Census Area.",
        "zip rejects non-digits (except one ZIP+4 hyphen) and values containing more than nine digits.",
    ],
    ParserType.DECIMAL: [
        "expected_data_type must include precision and scale; Spark precision is limited to 38.",
        "Spark rounds source values with excess scale to the configured decimal scale.",
    ],
    ParserType.DOUBLE: ["Use decimal(p,s) when exact base-10 representation matters."],
    ParserType.BOOLEAN: ["Unknown non-null tokens are parse errors, not false."],
    ParserType.DATE: ["Spark datetime patterns are not Python strptime patterns."],
    ParserType.TIMESTAMP: [
        "Timestamp interpretation follows the active Spark SQL session timezone."
    ],
}


def parser_description(parser_type: ParserType) -> dict[str, Any]:
    """Return a fresh machine-readable description for one parser type."""
    expected = ["decimal(p,s)"] if parser_type is ParserType.DECIMAL else [parser_type.value]
    return {
        "parser_type": parser_type.value,
        "expected_data_types": expected,
        "summary": _SUMMARIES[parser_type],
        "arguments": deepcopy([*_COMMON_ARGUMENTS, *_SPECIFIC_ARGUMENTS[parser_type]]),
        "key_behaviors": [
            "Whitespace collapse, trim, and empty-to-null run before parser-specific conversion.",
            "Null markers match only when replace_null_markers is true.",
            "Parse errors are resolved before zero and final-null handling.",
            "fail mode raises only when Spark materializes the failing silver expression; projection pruning can skip it.",
            "All execution uses native Spark expressions; no Python UDF is used.",
            *_SPECIFIC_BEHAVIORS.get(parser_type, []),
        ],
        "gotchas": deepcopy(_SPECIFIC_GOTCHAS.get(parser_type, [])),
    }


def config_description() -> dict[str, Any]:
    """Return the top-level, global, and column authoring contract."""
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
                default=PARSER_DEFAULTS["globals"]["null_marker_case_sensitive"],
                description="Default exact-case null matching behavior.",
            ),
            _argument(
                "true_values",
                default=PARSER_DEFAULTS["globals"]["true_values"],
                description="Global true-token list.",
            ),
            _argument(
                "false_values",
                default=PARSER_DEFAULTS["globals"]["false_values"],
                description="Global false-token list.",
            ),
            _argument(
                "boolean_case_sensitive",
                default=PARSER_DEFAULTS["globals"]["boolean_case_sensitive"],
                description="Whether global Boolean-token matching requires exact case.",
            ),
        ],
        "column_arguments": [
            _argument(
                "source_column_name",
                required=True,
                description="Exact top-level bronze source name; missing input warns and yields null/default.",
            ),
            _argument(
                "silver_column_name",
                required=True,
                description="Required, unique output name in parsed_df.",
            ),
            _argument(
                "expected_data_type",
                required=True,
                description="Exact target Spark type; validated against parser.type.",
            ),
            _argument(
                "parser", required=True, description="Scalar parser type or parser option mapping."
            ),
        ],
    }
