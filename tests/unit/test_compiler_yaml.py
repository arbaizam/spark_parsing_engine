"""Focused tests for the strict YAML authoring contract."""

from datetime import datetime, timedelta, timezone
from decimal import Decimal
from pathlib import Path
from types import MappingProxyType
from typing import Any

import pytest
import yaml

from spark_parser import (
    PARSER_DEFAULTS,
    BooleanValuesMode,
    ChildErrorMode,
    CompilationError,
    NullMarkersMode,
    ParseErrorMode,
    ParserConfig,
    ParserConfigSerializer,
    ParserType,
    StringFormat,
    YamlParserConfigCompiler,
    parse_spark_data_type,
)
from spark_parser.defaults import (
    DEFAULT_DATE_FORMATS,
    DEFAULT_TIMESTAMP_FORMATS,
    DEFAULT_TIMESTAMP_NTZ_FORMATS,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
TEST_CONFIG_PATH = REPO_ROOT / "tests" / "fixtures" / "test_config.yaml"


def _compile_default(
    expected_data_type: str,
    value: Any,
    **parser_options: Any,
) -> ParserConfig:
    """Compile one non-nullable column with a typed default through the public mapping API."""
    parser_type = expected_data_type.split("<", 1)[0].split("(", 1)[0]
    return YamlParserConfigCompiler().compile_mapping(
        {
            "parser_config_id": "default_test",
            "parser_config_name": "Default Test",
            "version": "1",
            "columns": [
                {
                    "source_column_name": "value",
                    "target_column_name": "Value",
                    "expected_data_type": expected_data_type,
                    "parser": {
                        "type": parser_type,
                        "is_nullable": False,
                        "default_on_null": value,
                        **parser_options,
                    },
                }
            ],
        }
    )


def test_recursive_spark_datatype_grammar_is_canonical() -> None:
    parsed = parse_spark_data_type(
        " STRUCT<`Display Name`: STRING, values: ARRAY<DECIMAL(18, 2)>, attrs: MAP<STRING, INT>> "
    )

    assert parsed.canonical == (
        "struct<`Display Name`:string,values:array<decimal(18,2)>,attrs:map<string,integer>>"
    )
    assert parsed.fields[1].data_type.element_type is not None


def test_complex_parsers_compile_recursively_and_round_trip() -> None:
    compiler = YamlParserConfigCompiler()
    config = compiler.compile_text(
        """
parser_config_id: complex
parser_config_name: Complex
version: "1"
columns:
  - source_column_name: names
    target_column_name: Names
    expected_data_type: array<string>
    parser:
      type: array
      element_parser: {type: string, format: upper}
      on_element_error: drop
      distinct: true
  - source_column_name: object
    target_column_name: Object
    expected_data_type: struct<name:string,scores:array<integer>>
    parser:
      type: struct
      fields:
        - {source_field_name: raw_name, target_field_name: name, parser: string}
        - source_field_name: raw_scores
          target_field_name: scores
          parser: {type: array, element_parser: integer, on_element_error: null}
  - source_column_name: attributes
    target_column_name: Attributes
    expected_data_type: map<string,decimal(8,2)>
    parser: {type: map, value_parser: decimal, on_value_error: drop}
"""
    )

    assert [column.parser.parser_type for column in config.columns] == [
        ParserType.ARRAY,
        ParserType.STRUCT,
        ParserType.MAP,
    ]
    struct_options = config.columns[1].parser
    assert struct_options.field_parsers[1].parser.element_parser is not None
    serializer = ParserConfigSerializer()
    payload = serializer.to_mapping(config)
    assert serializer.canonical_json(
        compiler.compile_mapping(payload)
    ) == serializer.canonical_json(config)


@pytest.mark.parametrize(
    ("expected_data_type", "parser_yaml", "message"),
    [
        ("map<integer,string>", "{type: map, value_parser: string}", "map<string"),
        (
            "array<array<string>>",
            "{type: array, input_format: delimited, delimiter: ',', element_parser: array}",
            "scalar element",
        ),
        (
            "array<string>",
            "{type: array, input_format: delimited, element_parser: string}",
            "delimiter",
        ),
        (
            "struct<a:string,b:string>",
            "{type: struct, fields: [{source_field_name: a, target_field_name: a, parser: string}]}",
            "missing field",
        ),
        (
            "array<map<string,string>>",
            "{type: array, element_parser: {type: map, value_parser: string}, distinct: true}",
            "non-comparable",
        ),
        ("variant", "string", "Unsupported datatype"),
    ],
)
def test_invalid_complex_contracts_fail_compilation(
    expected_data_type: str,
    parser_yaml: str,
    message: str,
) -> None:
    with pytest.raises(CompilationError, match=message):
        YamlParserConfigCompiler().compile_text(
            f"""
parser_config_id: invalid
parser_config_name: Invalid
version: "1"
columns:
  - source_column_name: value
    target_column_name: Value
    expected_data_type: {expected_data_type}
    parser: {parser_yaml}
"""
        )


@pytest.mark.parametrize(
    ("data_type", "default_value", "extra_option", "message"),
    [
        ("byte", "128", "", "does not fit byte"),
        ("short", "32768", "", "does not fit short"),
        ("binary", "GG", "encoding: hex", "valid hex"),
        ("binary", "not base64!", "", "valid base64"),
    ],
)
def test_new_scalar_defaults_are_strictly_validated(
    data_type: str,
    default_value: str,
    extra_option: str,
    message: str,
) -> None:
    with pytest.raises(CompilationError, match=message):
        YamlParserConfigCompiler().compile_text(
            f"""
parser_config_id: invalid_default
parser_config_name: Invalid Default
version: "1"
columns:
  - source_column_name: value
    target_column_name: Value
    expected_data_type: {data_type}
    parser:
      type: {data_type}
      is_nullable: false
      default_on_null: {default_value}
      {extra_option}
"""
        )


def test_repository_example_compiles_with_resolved_options() -> None:
    config = YamlParserConfigCompiler().compile_path(TEST_CONFIG_PATH)

    assert config.parser_config_id == "bronze_positions_to_target"
    assert config.version == "1.0.0"
    assert [column.expected_data_type for column in config.columns] == [
        "string",
        "integer",
        "decimal(38,32)",
        "date",
    ]
    string_options = config.columns[0].parser
    assert string_options.string_format is StringFormat.PASCAL
    assert string_options.null_markers == ("NA", "Null", "N/A")
    assert string_options.null_markers_mode is NullMarkersMode.EXTEND
    integer_options = config.columns[1].parser
    assert integer_options.default_on_null == -1
    assert integer_options.zero_is_valid is False


def test_all_supported_parsers_compile() -> None:
    config = YamlParserConfigCompiler().compile_path(REPO_ROOT / "examples" / "all_parsers.yaml")

    assert [column.parser.parser_type for column in config.columns] == list(ParserType)
    assert config.columns[3].parser.on_parse_error is ParseErrorMode.FAIL
    assert config.columns[3].parser.audit is False


def test_reference_example_columns_inherit_global_null_markers() -> None:
    """Prevent the reference template from silently replacing configured global markers."""
    example = (REPO_ROOT / "examples" / "all_parsers.yaml").read_text(encoding="utf-8")
    example = example.replace("  null_markers: []", "  null_markers: [NA]", 1)

    config = YamlParserConfigCompiler().compile_text(example)

    assert all(column.parser.null_markers == ("NA",) for column in config.columns)


def test_safe_defaults_are_explicit() -> None:
    config = YamlParserConfigCompiler().compile_text(
        """
parser_config_id: simple
parser_config_name: Simple
version: "1"
columns:
  - source_column_name: raw_name
    target_column_name: RawName
    expected_data_type: string
    parser: string
"""
    )

    options = config.columns[0].parser
    assert options.trim_whitespace is True
    assert options.collapse_whitespace is True
    assert options.empty_is_null is True
    assert options.string_format is None
    assert options.replace_null_markers is False
    assert options.is_nullable is True
    assert options.on_parse_error is ParseErrorMode.FAIL
    assert options.audit is False
    assert PARSER_DEFAULTS["common"]["collapse_whitespace"] is True
    assert PARSER_DEFAULTS["date"]["formats"] == DEFAULT_DATE_FORMATS
    assert PARSER_DEFAULTS["timestamp"]["formats"] == DEFAULT_TIMESTAMP_FORMATS
    assert PARSER_DEFAULTS["timestamp_ntz"]["formats"] == DEFAULT_TIMESTAMP_NTZ_FORMATS


@pytest.mark.parametrize("value", ["1.0e+300", "1.0e-100"])
def test_float_defaults_must_survive_spark_float32_narrowing(value: str) -> None:
    """Reject authored float defaults that Spark would silently turn into infinity or zero."""
    with pytest.raises(CompilationError, match="float32"):
        YamlParserConfigCompiler().compile_text(
            f"""
parser_config_id: float_range
parser_config_name: Float Range
version: "1"
columns:
  - source_column_name: value
    target_column_name: Value
    expected_data_type: float
    parser:
      type: float
      on_parse_error: default
      default_on_error: {value}
"""
        )

    # Spark double uses the same binary64 width as Python float, so the large finite value remains
    # valid there. This assertion prevents the float32 guard from accidentally narrowing doubles.
    if value == "1.0e+300":
        config = YamlParserConfigCompiler().compile_text(
            f"""
parser_config_id: double_range
parser_config_name: Double Range
version: "1"
columns:
  - source_column_name: value
    target_column_name: Value
    expected_data_type: double
    parser:
      type: double
      on_parse_error: default
      default_on_error: {value}
"""
        )
        assert config.columns[0].parser.default_on_error == 1.0e300


def test_timestamp_ntz_defaults_reject_timezone_offsets() -> None:
    """Keep local wall-clock defaults independent of the Spark session timezone."""
    invalid = """
parser_config_id: timestamp_default
parser_config_name: Timestamp Default
version: "1"
columns:
  - source_column_name: value
    target_column_name: Value
    expected_data_type: timestamp_ntz
    parser:
      type: timestamp_ntz
      is_nullable: false
      default_on_null: "2026-01-01T00:00:00+05:00"
"""
    with pytest.raises(CompilationError, match="must not include a timezone offset"):
        YamlParserConfigCompiler().compile_text(invalid)

    valid = invalid.replace("timestamp_ntz", "timestamp")
    config = YamlParserConfigCompiler().compile_text(valid)
    assert config.columns[0].parser.default_on_null.utcoffset() is not None


def test_timestamp_ntz_formats_reject_unquoted_timezone_fields() -> None:
    compiler = YamlParserConfigCompiler()
    for zone_field in "VvzOXxZ":
        with pytest.raises(
            CompilationError,
            match="timestamp_ntz must not contain unquoted timezone or offset",
        ):
            compiler.compile_text(
                f"""
parser_config_id: timestamp_ntz_zone
parser_config_name: Timestamp NTZ Zone
version: "1"
columns:
  - source_column_name: value
    target_column_name: Value
    expected_data_type: timestamp_ntz
    parser:
      type: timestamp_ntz
      formats: ["yyyy-MM-dd HH:mm:ss{zone_field}"]
"""
            )

    quoted = compiler.compile_text(
        """
parser_config_id: timestamp_ntz_quoted_zone
parser_config_name: Timestamp NTZ Quoted Zone
version: "1"
columns:
  - source_column_name: value
    target_column_name: Value
    expected_data_type: timestamp_ntz
    parser:
      type: timestamp_ntz
      formats: ["yyyy-MM-dd 'zone V' HH:mm:ss"]
"""
    )
    assert quoted.columns[0].parser.formats == ("yyyy-MM-dd 'zone V' HH:mm:ss",)


@pytest.mark.parametrize(
    ("format_name", "expected_format"),
    [
        ("title", StringFormat.TITLE),
        ("title_business_v1", StringFormat.TITLE_BUSINESS_V1),
        ("interest_rate_index_v1", StringFormat.INTEREST_RATE_INDEX_V1),
        ("state_us", StringFormat.STATE_US),
    ],
)
def test_display_string_formats_compile(
    format_name: str,
    expected_format: StringFormat,
) -> None:
    """Keep display-oriented formats distinct from identifier-oriented Pascal casing."""
    config = YamlParserConfigCompiler().compile_text(
        f"""
parser_config_id: display_format
parser_config_name: Display Format
version: "1"
columns:
  - source_column_name: raw_value
    target_column_name: DisplayValue
    expected_data_type: string
    parser:
      type: string
      format: {format_name}
"""
    )

    assert config.columns[0].parser.string_format is expected_format


def test_preserve_error_modes_compile_only_for_string_positions() -> None:
    """Allow exact raw fallback wherever—and only where—the typed result position is string."""
    config = YamlParserConfigCompiler().compile_text(
        """
parser_config_id: preserve_strings
parser_config_name: Preserve Strings
version: "1"
columns:
  - source_column_name: state
    target_column_name: State
    expected_data_type: string
    parser: {type: string, format: state_us, on_parse_error: preserve}
  - source_column_name: profile
    target_column_name: Profile
    expected_data_type: struct<state:string>
    parser:
      type: struct
      fields:
        - source_field_name: state
          target_field_name: state
          parser: {type: string, format: state_us, on_parse_error: preserve}
  - source_column_name: states
    target_column_name: States
    expected_data_type: array<string>
    parser:
      type: array
      element_parser: {type: string, format: state_us}
      on_element_error: preserve
  - source_column_name: state_map
    target_column_name: StateMap
    expected_data_type: map<string,string>
    parser:
      type: map
      value_parser: {type: string, format: state_us}
      on_value_error: preserve
"""
    )

    assert config.columns[0].parser.on_parse_error is ParseErrorMode.PRESERVE
    assert (
        config.columns[1].parser.field_parsers[0].parser.on_parse_error is ParseErrorMode.PRESERVE
    )
    assert config.columns[2].parser.on_element_error is ChildErrorMode.PRESERVE
    assert config.columns[3].parser.on_value_error is ChildErrorMode.PRESERVE
    canonical = ParserConfigSerializer().to_mapping(config)
    assert canonical["columns"][0]["parser"]["on_parse_error"] == "preserve"
    assert canonical["columns"][2]["parser"]["on_element_error"] == "preserve"
    assert canonical["columns"][3]["parser"]["on_value_error"] == "preserve"


@pytest.mark.parametrize(
    ("fragment", "message"),
    [
        ("expected_data_type: decimal(18,32)\n    parser: decimal", "Decimal scale"),
        ("expected_data_type: integer\n    parser: string", "incompatible"),
        (
            "expected_data_type: integer\n    parser:\n"
            "      type: integer\n      is_nullable: false",
            "requires default_on_null",
        ),
        (
            "expected_data_type: integer\n    parser:\n      type: integer\n"
            "      is_nullable: false\n      default_on_null: 0\n      zero_is_valid: false",
            "rejects zero",
        ),
        (
            "expected_data_type: integer\n    parser:\n      type: integer\n"
            "      on_parse_error: default\n      default_on_error: 0\n"
            "      zero_is_valid: false",
            "default_on_error",
        ),
        (
            "expected_data_type: integer\n    parser:\n"
            "      type: integer\n      on_parse_error: preserve",
            "requires a string parser",
        ),
        (
            "expected_data_type: array<integer>\n    parser:\n"
            "      type: array\n      element_parser: integer\n"
            "      on_element_error: preserve",
            "requires a string child parser",
        ),
        (
            "expected_data_type: map<string,decimal(8,2)>\n    parser:\n"
            "      type: map\n      value_parser: decimal\n"
            "      on_value_error: preserve",
            "requires a string child parser",
        ),
    ],
)
def test_invalid_contracts_fail(fragment: str, message: str) -> None:
    yaml_text = f"""
parser_config_id: invalid
parser_config_name: Invalid
version: "1"
columns:
  - source_column_name: value
    target_column_name: Value
    {fragment}
"""
    with pytest.raises(CompilationError, match=message):
        YamlParserConfigCompiler().compile_text(yaml_text)


def test_decimal_defaults_are_exact() -> None:
    config = YamlParserConfigCompiler().compile_text(
        """
parser_config_id: decimal_default
parser_config_name: Decimal Default
version: "1"
columns:
  - source_column_name: amount
    target_column_name: Amount
    expected_data_type: decimal(8,2)
    parser:
      type: decimal
      on_parse_error: default
      default_on_error: "1.25"
"""
    )

    assert config.columns[0].parser.default_on_error == Decimal("1.25")


def test_decimal_type_allows_authoring_whitespace_and_canonicalizes() -> None:
    config = YamlParserConfigCompiler().compile_text(
        """
parser_config_id: spaced_decimal
parser_config_name: Spaced Decimal
version: "1"
columns:
  - source_column_name: amount
    target_column_name: Amount
    expected_data_type: decimal(18, 2)
    parser: decimal
"""
    )

    assert config.columns[0].expected_data_type == "decimal(18,2)"


def test_duplicate_yaml_keys_fail() -> None:
    with pytest.raises(CompilationError, match="Duplicate YAML key") as exc_info:
        YamlParserConfigCompiler().compile_text(
            """
parser_config_id: duplicate
parser_config_id: duplicate_again
parser_config_name: Duplicate
version: "1"
columns: []
"""
        )
    assert "line 3, column 1" in str(exc_info.value)


def test_yaml_merge_keys_fail_with_an_actionable_message() -> None:
    """Explain the deliberate merge-key restriction instead of leaking a PyYAML tag error."""
    with pytest.raises(CompilationError, match=r"YAML merge keys \(<<\) are not supported"):
        YamlParserConfigCompiler().compile_text(
            """
parser_config_id: merge_key
parser_config_name: Merge Key
version: "1"
shared: &shared
  type: string
columns:
  - source_column_name: value
    target_column_name: Value
    expected_data_type: string
    parser:
      <<: *shared
"""
        )


def test_target_names_are_unique_but_sources_may_repeat() -> None:
    compiler = YamlParserConfigCompiler()
    config = compiler.compile_text(
        """
parser_config_id: repeated_source
parser_config_name: Repeated Source
version: "1"
columns:
  - source_column_name: raw_value
    target_column_name: RawValue
    expected_data_type: string
    parser: string
  - source_column_name: raw_value
    target_column_name: RawValueUpper
    expected_data_type: string
    parser:
      type: string
      format: upper
"""
    )

    assert [column.source_column_name for column in config.columns] == [
        "raw_value",
        "raw_value",
    ]
    with pytest.raises(CompilationError, match="Duplicate target_column_name"):
        compiler.compile_text(
            """
parser_config_id: duplicate_target
parser_config_name: Duplicate Target
version: "1"
columns:
  - source_column_name: first
    target_column_name: Value
    expected_data_type: string
    parser: string
  - source_column_name: second
    target_column_name: Value
    expected_data_type: string
    parser: string
"""
        )


def test_boolean_globals_inherit_extend_and_validate_case() -> None:
    config = YamlParserConfigCompiler().compile_text(
        """
parser_config_id: booleans
parser_config_name: Boolean Vocabularies
version: "1"
globals:
  true_values: ["true", Y]
  false_values: ["false", N]
  boolean_case_sensitive: false
columns:
  - source_column_name: active
    target_column_name: IsActive
    expected_data_type: boolean
    parser: boolean
  - source_column_name: approved
    target_column_name: IsApproved
    expected_data_type: boolean
    parser:
      type: boolean
      true_values: [approved]
      false_values: [rejected]
      boolean_values_mode: extend
"""
    )

    assert config.columns[0].parser.true_values == ("true", "Y")
    assert config.columns[1].parser.true_values == ("true", "Y", "approved")
    assert config.columns[1].parser.boolean_values_mode is BooleanValuesMode.EXTEND

    with pytest.raises(CompilationError, match="overlap"):
        YamlParserConfigCompiler().compile_text(
            """
parser_config_id: overlap
parser_config_name: Overlap
version: "1"
globals:
  true_values: ["YES"]
  false_values: ["yes"]
  boolean_case_sensitive: false
columns:
  - source_column_name: value
    target_column_name: Value
    expected_data_type: boolean
    parser: boolean
"""
        )


def test_boolean_one_sided_overrides_and_runtime_lower_contract() -> None:
    config = YamlParserConfigCompiler().compile_text(
        """
parser_config_id: one_sided
parser_config_name: One Sided Boolean
version: "1"
globals:
  true_values: ["true", Y]
  false_values: ["false", N]
  boolean_case_sensitive: false
columns:
  - source_column_name: replace_value
    target_column_name: ReplaceValue
    expected_data_type: boolean
    parser:
      type: boolean
      true_values: [approved]
  - source_column_name: extend_value
    target_column_name: ExtendValue
    expected_data_type: boolean
    parser:
      type: boolean
      false_values: [rejected]
      boolean_values_mode: extend
  - source_column_name: unicode_value
    target_column_name: UnicodeValue
    expected_data_type: boolean
    parser:
      type: boolean
      true_values: ["ß"]
      false_values: [SS]
"""
    )

    replace = config.columns[0].parser
    assert replace.true_values == ("approved",)
    assert replace.false_values == ("false", "N")
    extend = config.columns[1].parser
    assert extend.true_values == ("true", "Y")
    assert extend.false_values == ("false", "N", "rejected")
    assert config.columns[2].parser.true_values == ("ß",)

    # Non-ASCII case mapping belongs to Spark. Even a familiar pair is accepted by the Spark-free
    # compiler and marked for runtime overlap validation rather than decided with Python's tables.
    deferred = YamlParserConfigCompiler().compile_text(
        """
parser_config_id: deferred_unicode_overlap
parser_config_name: Deferred Unicode Overlap
version: "1"
globals:
  true_values: ["Ä"]
  false_values: ["ä"]
  boolean_case_sensitive: false
columns:
  - source_column_name: value
    target_column_name: Value
    expected_data_type: boolean
    parser: boolean
"""
    )
    assert deferred.columns[0].parser.true_values == ("Ä",)
    assert deferred.columns[0].parser.false_values == ("ä",)


def test_unknown_column_keys_and_invalid_mapping_inputs_have_targeted_errors() -> None:
    compiler = YamlParserConfigCompiler()
    with pytest.raises(CompilationError, match="unsupported keys"):
        compiler.compile_text(
            """
parser_config_id: unknown_keys
parser_config_name: Unknown Keys
version: "1"
columns:
  - source_column_name: value
    target_column_name: Value
    expected_data_type: string
    parser: string
    unexpected_option: true
"""
        )
    with pytest.raises(CompilationError, match="parser config must be a mapping"):
        compiler.compile_mapping("not a mapping")  # type: ignore[arg-type]
    with pytest.raises(CompilationError, match="keys must be strings"):
        compiler.compile_mapping({1: "invalid"})  # type: ignore[dict-item]


def test_metadata_is_trimmed_but_source_and_target_names_are_preserved() -> None:
    compiler = YamlParserConfigCompiler()
    serializer = ParserConfigSerializer()
    config = compiler.compile_text(
        """
parser_config_id: "  trimmed  "
parser_config_name: "  Trimmed Name  "
version: "  1  "
columns:
  - source_column_name: "  raw_value  "
    target_column_name: "  Value  "
    expected_data_type: " string "
    parser: string
  - source_column_name: "  raw_object  "
    target_column_name: "  Object  "
    expected_data_type: "struct<`  Field  `:string>"
    parser:
      type: struct
      fields:
        - source_field_name: "  raw_field  "
          target_field_name: "  Field  "
          parser: string
"""
    )

    assert config.parser_config_id == "trimmed"
    assert config.parser_config_name == "Trimmed Name"
    assert config.version == "1"
    assert config.columns[0].source_column_name == "  raw_value  "
    assert config.columns[0].target_column_name == "  Value  "
    field = config.columns[1].parser.field_parsers[0]
    assert field.source_field_name == "  raw_field  "
    assert field.target_field_name == "  Field  "

    resolved = serializer.to_mapping(config)
    assert resolved["columns"][0]["source_column_name"] == "  raw_value  "
    assert resolved["columns"][1]["parser"]["fields"][0]["source_field_name"] == (
        "  raw_field  "
    )
    recompiled = compiler.compile_mapping(resolved)
    assert serializer.content_hash(recompiled) == serializer.content_hash(config)
    changed_names = serializer.to_mapping(config)
    changed_names["columns"][0]["source_column_name"] = "raw_value"
    changed_names["columns"][0]["target_column_name"] = "Value"
    assert serializer.content_hash(compiler.compile_mapping(changed_names)) != (
        serializer.content_hash(config)
    )


def test_complex_parsers_resolve_collapse_whitespace_to_false() -> None:
    config = YamlParserConfigCompiler().compile_text(
        """
parser_config_id: complex_normalization
parser_config_name: Complex Normalization
version: "1"
columns:
  - source_column_name: names
    target_column_name: Names
    expected_data_type: array<string>
    parser:
      type: array
      collapse_whitespace: true
      element_parser: string
  - source_column_name: object
    target_column_name: Object
    expected_data_type: struct<name:string>
    parser:
      type: struct
      fields:
        - {source_field_name: name, target_field_name: name, parser: string}
  - source_column_name: attributes
    target_column_name: Attributes
    expected_data_type: map<string,string>
    parser: {type: map, value_parser: string}
  - source_column_name: label
    target_column_name: Label
    expected_data_type: string
    parser: string
"""
    )

    array_options, struct_options, map_options, string_options = (
        column.parser for column in config.columns
    )
    assert array_options.collapse_whitespace is False
    assert struct_options.collapse_whitespace is False
    assert map_options.collapse_whitespace is False
    assert array_options.element_parser is not None
    assert array_options.element_parser.parser.collapse_whitespace is True
    assert struct_options.field_parsers[0].parser.collapse_whitespace is True
    assert string_options.collapse_whitespace is True

    payload = ParserConfigSerializer().to_mapping(config)
    assert payload["columns"][0]["parser"]["collapse_whitespace"] is False
    assert payload["columns"][3]["parser"]["collapse_whitespace"] is True


@pytest.mark.parametrize(
    ("alias", "canonical"),
    [
        ("tinyint", "byte"),
        ("smallint", "short"),
        ("int", "integer"),
        ("bigint", "long"),
        ("real", "float"),
        ("bool", "boolean"),
        ("timestamp_ltz", "timestamp"),
    ],
)
def test_datatype_and_parser_aliases_share_one_table(alias: str, canonical: str) -> None:
    assert parse_spark_data_type(alias).parser_type.value == canonical
    config = YamlParserConfigCompiler().compile_text(
        f"""
parser_config_id: alias
parser_config_name: Alias
version: "1"
columns:
  - source_column_name: value
    target_column_name: Value
    expected_data_type: {alias}
    parser: {alias}
"""
    )

    assert config.columns[0].expected_data_type == canonical
    assert config.columns[0].parser.parser_type.value == canonical


@pytest.mark.parametrize("alias", ["dec", "numeric"])
def test_decimal_datatype_and_parser_aliases_share_one_table(alias: str) -> None:
    assert parse_spark_data_type(f"{alias}(5,2)").canonical == "decimal(5,2)"
    config = YamlParserConfigCompiler().compile_text(
        f"""
parser_config_id: decimal_alias
parser_config_name: Decimal Alias
version: "1"
columns:
  - source_column_name: value
    target_column_name: Value
    expected_data_type: {alias}(5,2)
    parser: {alias}
"""
    )

    assert config.columns[0].expected_data_type == "decimal(5,2)"
    assert config.columns[0].parser.parser_type is ParserType.DECIMAL


def test_ddl_nesting_has_a_deterministic_limit() -> None:
    deepest_supported = "array<" * 64 + "string" + ">" * 64
    assert parse_spark_data_type(deepest_supported).canonical == deepest_supported

    too_deep = "array<" * 65 + "string" + ">" * 65
    with pytest.raises(CompilationError, match="maximum depth of 64"):
        parse_spark_data_type(too_deep)


def test_yaml_authoring_supports_the_full_logical_datatype_depth() -> None:
    expected_data_type = "string"
    nested_parser: Any = "string"
    for _ in range(64):
        expected_data_type = f"struct<value:{expected_data_type}>"
        nested_parser = {
            "type": "struct",
            "fields": [
                {
                    "source_field_name": "value",
                    "target_field_name": "value",
                    "parser": nested_parser,
                }
            ],
        }
    payload = {
        "parser_config_id": "deep_yaml",
        "parser_config_name": "Deep YAML",
        "version": "1",
        "columns": [
            {
                "source_column_name": "value",
                "target_column_name": "Value",
                "expected_data_type": expected_data_type,
                "parser": nested_parser,
            }
        ],
    }

    config = YamlParserConfigCompiler().compile_text(yaml.safe_dump(payload, sort_keys=False))
    assert config.columns[0].expected_data_type == expected_data_type


def test_decimal_parameters_use_bounded_ascii_integer_tokens() -> None:
    assert parse_spark_data_type("decimal(00038,00002)").canonical == "decimal(38,2)"

    for invalid in ("decimal(٣٨,2)", "decimal(²,2)"):
        with pytest.raises(CompilationError, match="decimal precision"):
            parse_spark_data_type(invalid)

    with pytest.raises(CompilationError, match="Decimal precision"):
        parse_spark_data_type(f"decimal({'9' * 5_000},2)")


def test_ddl_unicode_fields_round_trip_and_unpaired_surrogates_fail() -> None:
    parsed = parse_spark_data_type("struct<`emoji😀`:string>")
    assert parse_spark_data_type(parsed.canonical) == parsed

    with pytest.raises(CompilationError, match="well-formed Unicode"):
        parse_spark_data_type("struct<`\ud800`:string>")


def test_yaml_key_errors_include_source_location_and_depth_is_bounded() -> None:
    compiler = YamlParserConfigCompiler()
    with pytest.raises(CompilationError, match=r"hashable.*line 1, column 3"):
        compiler.compile_text("? [a, b]\n: value\n")

    with pytest.raises(CompilationError, match="YAML nesting exceeds the maximum depth of 256"):
        compiler.compile_text("[" * 260 + "value" + "]" * 260)

    huge_integer = "9" * 5_000
    with pytest.raises(CompilationError):
        compiler.compile_text(
            f"""
parser_config_id: {huge_integer}
parser_config_name: Huge Integer
version: "1"
columns:
  - source_column_name: value
    target_column_name: Value
    expected_data_type: string
    parser: string
"""
        )


def test_compiler_rejects_unpaired_surrogates_in_all_string_boundaries() -> None:
    compiler = YamlParserConfigCompiler()
    escaped_metadata = """
parser_config_id: "\\uD800"
parser_config_name: Invalid Unicode
version: "1"
columns:
  - source_column_name: value
    target_column_name: Value
    expected_data_type: string
    parser: string
"""
    with pytest.raises(CompilationError, match="well-formed Unicode"):
        compiler.compile_text(escaped_metadata)

    with pytest.raises(CompilationError, match="well-formed Unicode"):
        _compile_default("string", "\ud800")

    with pytest.raises(CompilationError, match="well-formed Unicode"):
        _compile_default(
            "map<string,string>",
            {"\ud800": "value"},
            value_parser="string",
        )

    with pytest.raises(CompilationError, match="well-formed Unicode"):
        _compile_default(
            "array<string>",
            ["value"],
            input_format="delimited",
            delimiter="\ud800",
            element_parser="string",
        )


def test_compile_path_reports_malformed_utf8(tmp_path: Path) -> None:
    config_path = tmp_path / "invalid.yaml"
    config_path.write_bytes(b"parser_config_id: \xff\n")

    with pytest.raises(CompilationError, match="well-formed UTF-8"):
        YamlParserConfigCompiler().compile_path(config_path)


def test_complex_defaults_are_deeply_frozen_and_round_trip() -> None:
    config = _compile_default(
        "array<map<string,string>>",
        [{"key": "value"}],
        element_parser={"type": "map", "value_parser": "string"},
    )
    compiled_default = config.columns[0].parser.default_on_null

    assert isinstance(compiled_default, tuple)
    assert isinstance(compiled_default[0], MappingProxyType)
    with pytest.raises(AttributeError):
        compiled_default.append({})
    with pytest.raises(TypeError):
        compiled_default[0]["key"] = "mutated"

    serializer = ParserConfigSerializer()
    serialized = serializer.to_mapping(config)
    round_tripped = YamlParserConfigCompiler().compile_mapping(serialized)
    assert serializer.content_hash(round_tripped) == serializer.content_hash(config)
    serialized["columns"][0]["parser"]["default_on_null"][0]["key"] = "changed"
    assert compiled_default[0]["key"] == "value"


def test_expanded_default_size_limit_has_an_exact_round_trip_boundary() -> None:
    # The array container plus 9,999 scalar elements consumes the complete 10,000-node budget.
    exact_boundary = list(range(9_999))
    config = _compile_default(
        "array<integer>",
        exact_boundary,
        element_parser="integer",
    )
    serializer = ParserConfigSerializer()
    round_tripped = YamlParserConfigCompiler().compile_mapping(serializer.to_mapping(config))

    assert len(config.columns[0].parser.default_on_null) == 9_999
    assert serializer.content_hash(round_tripped) == serializer.content_hash(config)

    with pytest.raises(CompilationError, match=r"default_on_null\[9999\].*10,000 nodes"):
        _compile_default(
            "array<integer>",
            list(range(10_000)),
            element_parser="integer",
        )
    with pytest.raises(CompilationError, match=r"default_on_null\[9999\].*10,000 nodes"):
        _compile_default(
            "array<integer>",
            [None] * 10_000,
            element_parser="integer",
        )


def test_expanded_default_budget_counts_map_keys_and_values() -> None:
    oversized = {f"key{index}": index for index in range(5_000)}

    with pytest.raises(CompilationError, match=r'default_on_null\["key4999"\].*10,000 nodes'):
        _compile_default(
            "map<string,integer>",
            oversized,
            value_parser="integer",
        )


def test_shared_default_aliases_cannot_evade_the_expanded_size_limit() -> None:
    value: Any = 0
    data_type = "integer"
    parser: Any = "integer"
    for _ in range(6):
        # Only six unique list objects, but five aliases per level expand beyond 10,000 literals.
        value = [value] * 5
        data_type = f"array<{data_type}>"
        parser = {"type": "array", "element_parser": parser}

    with pytest.raises(
        CompilationError,
        match=r"default_on_null\[.*maximum expanded default size of 10,000 nodes",
    ):
        _compile_default(
            data_type,
            value,
            element_parser=parser["element_parser"],
        )


def test_complex_default_cycles_report_the_reference_path() -> None:
    direct_cycle: list[Any] = []
    direct_cycle.append(direct_cycle)
    with pytest.raises(CompilationError, match=r"default_on_null\[0\].*cyclic"):
        _compile_default(
            "array<array<integer>>",
            direct_cycle,
            element_parser={"type": "array", "element_parser": "integer"},
        )

    indirect_cycle: list[Any] = []
    middle = [indirect_cycle]
    indirect_cycle.append(middle)
    with pytest.raises(CompilationError, match=r"default_on_null\[0\]\[0\].*cyclic"):
        _compile_default(
            "array<array<array<integer>>>",
            indirect_cycle,
            element_parser={
                "type": "array",
                "element_parser": {"type": "array", "element_parser": "integer"},
            },
        )

    with pytest.raises(CompilationError, match=r"default_on_null\[0\].*cyclic"):
        YamlParserConfigCompiler().compile_text(
            """
parser_config_id: cyclic_alias
parser_config_name: Cyclic Alias
version: "1"
columns:
  - source_column_name: value
    target_column_name: Value
    expected_data_type: array<array<integer>>
    parser:
      type: array
      element_parser: {type: array, element_parser: integer}
      is_nullable: false
      default_on_null: &cycle [*cycle]
"""
        )


@pytest.mark.parametrize(
    ("data_type", "value", "element_parser", "expected_path"),
    [
        ("array<integer>", [1, "bad", 3], "integer", "default_on_null[1]"),
        (
            "array<array<integer>>",
            [[1], [2, "bad"]],
            {"type": "array", "element_parser": "integer"},
            "default_on_null[1][1]",
        ),
    ],
)
def test_array_default_errors_include_the_full_element_path(
    data_type: str,
    value: Any,
    element_parser: Any,
    expected_path: str,
) -> None:
    with pytest.raises(CompilationError) as exc_info:
        _compile_default(data_type, value, element_parser=element_parser)

    assert f"{expected_path} for integer must be an integer" in str(exc_info.value)


@pytest.mark.parametrize(
    ("field_name", "expected_path"),
    [
        ("a.b", 'default_on_null["a.b"]'),
        ("line\nbreak", 'default_on_null["line\\nbreak"]'),
    ],
)
def test_struct_default_errors_quote_unsafe_field_paths(
    field_name: str,
    expected_path: str,
) -> None:
    expected_data_type = f"struct<`{field_name}`:integer>"
    with pytest.raises(CompilationError) as exc_info:
        _compile_default(
            expected_data_type,
            {field_name: "bad"},
            fields=[
                {
                    "source_field_name": field_name,
                    "target_field_name": field_name,
                    "parser": "integer",
                }
            ],
        )

    message = str(exc_info.value)
    assert f"{expected_path} for integer must be an integer" in message
    assert "\n" not in message


def test_binary_default_errors_quote_unsafe_struct_field_paths() -> None:
    with pytest.raises(CompilationError) as exc_info:
        _compile_default(
            "struct<`a.b`:binary>",
            {"a.b": "GG"},
            fields=[
                {
                    "source_field_name": "a.b",
                    "target_field_name": "a.b",
                    "parser": {"type": "binary", "encoding": "hex"},
                }
            ],
        )

    assert 'default_on_null["a.b"] is not valid hex binary text' in str(exc_info.value)


def test_decimal_defaults_are_canonical_at_the_declared_scale() -> None:
    serializer = ParserConfigSerializer()
    zero_configs = [_compile_default("decimal(2,2)", value) for value in ("0", "-0", "0E+999")]
    assert {config.columns[0].parser.default_on_null for config in zero_configs} == {
        Decimal("0.00")
    }
    assert len({serializer.content_hash(config) for config in zero_configs}) == 1

    equivalent_configs = [
        _compile_default("decimal(8,2)", value) for value in ("1.2", "1.20", "12E-1")
    ]
    assert all(
        config.columns[0].parser.default_on_null == Decimal("1.20") for config in equivalent_configs
    )
    assert len({serializer.content_hash(config) for config in equivalent_configs}) == 1

    scientific = _compile_default("decimal(8,2)", "1E+3")
    assert scientific.columns[0].parser.default_on_null == Decimal("1000.00")
    resolved = serializer.to_mapping(scientific)
    assert resolved["columns"][0]["parser"]["default_on_null"] == "1000.00"
    recompiled = YamlParserConfigCompiler().compile_mapping(resolved)
    assert serializer.content_hash(recompiled) == serializer.content_hash(scientific)
    trailing_zeroes = _compile_default("decimal(8,2)", "1.2300")
    assert str(trailing_zeroes.columns[0].parser.default_on_null) == "1.23"
    with pytest.raises(CompilationError, match=r"does not fit decimal\(8,2\)"):
        _compile_default("decimal(8,2)", "1.231")


@pytest.mark.parametrize("value", ["1_0", " 1.5 ", "\t1.5", "１２.５"])
def test_decimal_string_defaults_require_strict_ascii_numeric_text(value: str) -> None:
    with pytest.raises(CompilationError, match="for decimal must be numeric"):
        _compile_default("decimal(8,2)", value)


@pytest.mark.parametrize("value", ["20260830", "2026-W35-7", "2026-8-30", "2026-02-30"])
def test_date_string_defaults_use_the_package_iso_grammar(value: str) -> None:
    with pytest.raises(CompilationError, match="ISO YYYY-MM-DD"):
        _compile_default("date", value)


@pytest.mark.parametrize(
    "value",
    [
        "2026-08-30",
        "20260830T010203",
        "2026-08-30 01:02:03",
        "2026-08-30T01:02:03.1234567",
        "2026-08-30T01:02:03+0500",
        "2026-08-30T01:02:03+01:60",
        "2026-08-30T01:02:03+01:99",
        "2026-08-30T01:02:03+18:01",
        "2026-08-30T01:02:03+23:59",
        "2026-08-30T01:02:03+24:00",
    ],
)
def test_timestamp_string_defaults_use_the_package_iso_grammar(value: str) -> None:
    with pytest.raises(CompilationError, match="must use ISO"):
        _compile_default("timestamp", value)


@pytest.mark.parametrize(
    "value",
    [
        "2026-08-30 01:02:03",
        "2026-08-30t01:02:03Z",
        "2026-08-30T01:02:03.1234567",
    ],
)
def test_implicitly_tagged_yaml_timestamps_still_use_package_grammar(value: str) -> None:
    with pytest.raises(CompilationError, match="must use ISO"):
        YamlParserConfigCompiler().compile_text(
            f"""
parser_config_id: timestamp_default
parser_config_name: Timestamp Default
version: "1"
columns:
  - source_column_name: value
    target_column_name: Value
    expected_data_type: timestamp
    parser:
      type: timestamp
      is_nullable: false
      default_on_null: {value}
"""
        )


def test_strict_date_and_timestamp_defaults_accept_canonical_forms() -> None:
    date_config = _compile_default("date", "2026-08-30")
    assert date_config.columns[0].parser.default_on_null.isoformat() == "2026-08-30"

    timestamp_config = _compile_default("timestamp", "2026-08-30T01:02:03.123456Z")
    timestamp_default = timestamp_config.columns[0].parser.default_on_null
    assert timestamp_default.isoformat() == "2026-08-30T01:02:03.123456+00:00"

    maximum_offset = _compile_default("timestamp", "2026-08-30T01:02:03+18:00")
    assert maximum_offset.columns[0].parser.default_on_null.isoformat() == (
        "2026-08-29T07:02:03+00:00"
    )

    yaml_config = YamlParserConfigCompiler().compile_text(
        """
parser_config_id: timestamp_default
parser_config_name: Timestamp Default
version: "1"
columns:
  - source_column_name: value
    target_column_name: Value
    expected_data_type: timestamp
    parser:
      type: timestamp
      is_nullable: false
      default_on_null: 2026-08-30T01:02:03Z
"""
    )
    assert yaml_config.columns[0].parser.default_on_null == timestamp_default.replace(microsecond=0)


@pytest.mark.parametrize("parser_type", ["timestamp", "timestamp_ntz"])
@pytest.mark.parametrize("fraction_digits", range(1, 7))
def test_timestamp_defaults_accept_every_documented_fraction_width_on_all_python_versions(
    parser_type: str,
    fraction_digits: int,
) -> None:
    fraction = "123456"[:fraction_digits]
    suffix = "Z" if parser_type == "timestamp" else ""
    config = _compile_default(
        parser_type,
        f"2024-01-02T03:04:05.{fraction}{suffix}",
    )

    assert config.columns[0].parser.default_on_null.microsecond == int(fraction.ljust(6, "0"))


def test_datetime_defaults_require_round_trip_safe_timezone_offsets() -> None:
    invalid_offset = datetime(
        2026,
        8,
        30,
        1,
        2,
        3,
        tzinfo=timezone(timedelta(seconds=30)),
    )
    with pytest.raises(CompilationError, match="whole-minute timezone offset"):
        _compile_default("timestamp", invalid_offset)

    valid_offset = datetime(
        2026,
        8,
        30,
        1,
        2,
        3,
        tzinfo=timezone(timedelta(hours=5, minutes=45)),
    )
    config = _compile_default("timestamp", valid_offset)
    assert config.columns[0].parser.default_on_null == datetime(
        2026,
        8,
        29,
        19,
        17,
        3,
        tzinfo=timezone.utc,
    )
    serializer = ParserConfigSerializer()
    round_tripped = YamlParserConfigCompiler().compile_mapping(serializer.to_mapping(config))
    assert serializer.content_hash(round_tripped) == serializer.content_hash(config)


def test_equivalent_aware_timestamp_offsets_share_content_identity() -> None:
    offset_config = _compile_default(
        "timestamp",
        "2026-08-30T01:02:03.123456+05:45",
    )
    utc_config = _compile_default(
        "timestamp",
        "2026-08-29T19:17:03.123456Z",
    )
    serializer = ParserConfigSerializer()

    assert (
        offset_config.columns[0].parser.default_on_null
        == utc_config.columns[0].parser.default_on_null
    )
    assert serializer.content_hash(offset_config) == serializer.content_hash(utc_config)
    assert (
        serializer.to_mapping(offset_config)["columns"][0]["parser"]["default_on_null"]
        == "2026-08-29T19:17:03.123456+00:00"
    )


@pytest.mark.parametrize("value", ["", "F", "abc", "ABC12"])
def test_hex_defaults_follow_spark_unhex_grammar(value: str) -> None:
    config = _compile_default("binary", value, encoding="hex")
    assert config.columns[0].parser.default_on_null == value


@pytest.mark.parametrize("value", ["01 AF", "0x12", "Ａ", "é"])
def test_hex_defaults_reject_whitespace_prefixes_and_non_ascii(value: str) -> None:
    with pytest.raises(CompilationError, match="not valid hex binary text"):
        _compile_default("binary", value, encoding="hex")


def test_float_conversion_range_failures_are_compilation_errors() -> None:
    with pytest.raises(CompilationError, match="double must be finite"):
        _compile_default("double", 10**10_000)

    for parser_type in ("float", "double"):
        with pytest.raises(CompilationError, match="underflows to zero"):
            _compile_default(parser_type, Decimal("1E-10000"))


def test_duplicate_scans_report_each_name_once_in_sorted_order() -> None:
    with pytest.raises(CompilationError, match=r"duplicate fields: \['a', 'b'\]"):
        parse_spark_data_type("struct<b:string,a:string,b:string,a:string,b:string>")

    payload = {
        "parser_config_id": "duplicates",
        "parser_config_name": "Duplicates",
        "version": "1",
        "columns": [
            {
                "source_column_name": str(index),
                "target_column_name": name,
                "expected_data_type": "string",
                "parser": "string",
            }
            for index, name in enumerate(("b", "a", "b", "a", "b"))
        ],
    }
    with pytest.raises(
        CompilationError,
        match=r"Duplicate target_column_name values: \['a', 'b'\]",
    ):
        YamlParserConfigCompiler().compile_mapping(payload)

    with pytest.raises(CompilationError, match=r"duplicate source fields \['raw'\]"):
        YamlParserConfigCompiler().compile_mapping(
            {
                "parser_config_id": "struct_duplicates",
                "parser_config_name": "Struct Duplicates",
                "version": "1",
                "columns": [
                    {
                        "source_column_name": "object",
                        "target_column_name": "Object",
                        "expected_data_type": "struct<a:string,b:string>",
                        "parser": {
                            "type": "struct",
                            "fields": [
                                {
                                    "source_field_name": "raw",
                                    "target_field_name": "a",
                                    "parser": "string",
                                },
                                {
                                    "source_field_name": "raw",
                                    "target_field_name": "b",
                                    "parser": "string",
                                },
                            ],
                        },
                    }
                ],
            }
        )
