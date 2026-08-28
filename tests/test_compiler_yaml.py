"""Focused tests for the strict YAML authoring contract."""

from decimal import Decimal
from pathlib import Path

import pytest

from spark_parser import (
    PARSER_DEFAULTS,
    BooleanValuesMode,
    CompilationError,
    NullMarkersMode,
    ParseErrorMode,
    ParserConfigSerializer,
    ParserType,
    StringFormat,
    YamlParserConfigCompiler,
    parse_spark_data_type,
)

ROOT = Path(__file__).resolve().parents[1]


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
    silver_column_name: Names
    expected_data_type: array<string>
    parser:
      type: array
      element_parser: {type: string, format: upper}
      on_element_error: drop
      distinct: true
  - source_column_name: object
    silver_column_name: Object
    expected_data_type: struct<name:string,scores:array<integer>>
    parser:
      type: struct
      fields:
        - {source_field_name: raw_name, silver_field_name: name, parser: string}
        - source_field_name: raw_scores
          silver_field_name: scores
          parser: {type: array, element_parser: integer, on_element_error: null}
  - source_column_name: attributes
    silver_column_name: Attributes
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
            "{type: struct, fields: [{source_field_name: a, silver_field_name: a, parser: string}]}",
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
    silver_column_name: Value
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
    silver_column_name: Value
    expected_data_type: {data_type}
    parser:
      type: {data_type}
      is_nullable: false
      default_on_null: {default_value}
      {extra_option}
"""
        )


def test_repository_example_compiles_with_resolved_options() -> None:
    config = YamlParserConfigCompiler().compile_path(ROOT / "test_config.yaml")

    assert config.parser_config_id == "bronze_positions_to_silver"
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
    config = YamlParserConfigCompiler().compile_path(ROOT / "examples" / "all_parsers.yaml")

    assert [column.parser.parser_type for column in config.columns] == list(ParserType)
    assert config.columns[3].parser.on_parse_error is ParseErrorMode.FAIL
    assert config.columns[3].parser.audit is False


def test_safe_defaults_are_explicit() -> None:
    config = YamlParserConfigCompiler().compile_text(
        """
parser_config_id: simple
parser_config_name: Simple
version: "1"
columns:
  - source_column_name: raw_name
    silver_column_name: RawName
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
    assert PARSER_DEFAULTS["date"]["formats"] == ["yyyy-MM-dd"]


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
    ],
)
def test_invalid_contracts_fail(fragment: str, message: str) -> None:
    yaml_text = f"""
parser_config_id: invalid
parser_config_name: Invalid
version: "1"
columns:
  - source_column_name: value
    silver_column_name: Value
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
    silver_column_name: Amount
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
    silver_column_name: Amount
    expected_data_type: decimal(18, 2)
    parser: decimal
"""
    )

    assert config.columns[0].expected_data_type == "decimal(18,2)"


def test_duplicate_yaml_keys_fail() -> None:
    with pytest.raises(CompilationError, match="Duplicate YAML key"):
        YamlParserConfigCompiler().compile_text(
            """
parser_config_id: duplicate
parser_config_id: duplicate_again
parser_config_name: Duplicate
version: "1"
columns: []
"""
        )


def test_silver_names_are_unique_but_sources_may_repeat() -> None:
    compiler = YamlParserConfigCompiler()
    config = compiler.compile_text(
        """
parser_config_id: repeated_source
parser_config_name: Repeated Source
version: "1"
columns:
  - source_column_name: raw_value
    silver_column_name: RawValue
    expected_data_type: string
    parser: string
  - source_column_name: raw_value
    silver_column_name: RawValueUpper
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
    with pytest.raises(CompilationError, match="Duplicate silver_column_name"):
        compiler.compile_text(
            """
parser_config_id: duplicate_silver
parser_config_name: Duplicate Silver
version: "1"
columns:
  - source_column_name: first
    silver_column_name: Value
    expected_data_type: string
    parser: string
  - source_column_name: second
    silver_column_name: Value
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
    silver_column_name: IsActive
    expected_data_type: boolean
    parser: boolean
  - source_column_name: approved
    silver_column_name: IsApproved
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
    silver_column_name: Value
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
    silver_column_name: ReplaceValue
    expected_data_type: boolean
    parser:
      type: boolean
      true_values: [approved]
  - source_column_name: extend_value
    silver_column_name: ExtendValue
    expected_data_type: boolean
    parser:
      type: boolean
      false_values: [rejected]
      boolean_values_mode: extend
  - source_column_name: unicode_value
    silver_column_name: UnicodeValue
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


def test_legacy_keys_and_invalid_mapping_inputs_have_targeted_errors() -> None:
    compiler = YamlParserConfigCompiler()
    with pytest.raises(CompilationError, match="0.2.x keys"):
        compiler.compile_text(
            """
parser_config_id: legacy
parser_config_name: Legacy
version: "1"
columns:
  - column_name: value
    data_type: string
    parser: string
"""
        )
    with pytest.raises(CompilationError, match="parser config must be a mapping"):
        compiler.compile_mapping("not a mapping")  # type: ignore[arg-type]
    with pytest.raises(CompilationError, match="keys must be strings"):
        compiler.compile_mapping({1: "invalid"})  # type: ignore[dict-item]


def test_required_metadata_is_trimmed() -> None:
    config = YamlParserConfigCompiler().compile_text(
        """
parser_config_id: "  trimmed  "
parser_config_name: "  Trimmed Name  "
version: "  1  "
columns:
  - source_column_name: "  raw_value  "
    silver_column_name: "  Value  "
    expected_data_type: " string "
    parser: string
"""
    )

    assert config.parser_config_id == "trimmed"
    assert config.version == "1"
    assert config.columns[0].source_column_name == "raw_value"
    assert config.columns[0].silver_column_name == "Value"


def test_complex_parsers_resolve_collapse_whitespace_to_false() -> None:
    config = YamlParserConfigCompiler().compile_text(
        """
parser_config_id: complex_normalization
parser_config_name: Complex Normalization
version: "1"
columns:
  - source_column_name: names
    silver_column_name: Names
    expected_data_type: array<string>
    parser:
      type: array
      collapse_whitespace: true
      element_parser: string
  - source_column_name: object
    silver_column_name: Object
    expected_data_type: struct<name:string>
    parser:
      type: struct
      fields:
        - {source_field_name: name, silver_field_name: name, parser: string}
  - source_column_name: attributes
    silver_column_name: Attributes
    expected_data_type: map<string,string>
    parser: {type: map, value_parser: string}
  - source_column_name: label
    silver_column_name: Label
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
    silver_column_name: Value
    expected_data_type: {alias}
    parser: {alias}
"""
    )

    assert config.columns[0].expected_data_type == canonical
    assert config.columns[0].parser.parser_type.value == canonical
