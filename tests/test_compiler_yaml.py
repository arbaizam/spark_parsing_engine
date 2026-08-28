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
    ParserType,
    StringFormat,
    YamlParserConfigCompiler,
)

ROOT = Path(__file__).resolve().parents[1]


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
