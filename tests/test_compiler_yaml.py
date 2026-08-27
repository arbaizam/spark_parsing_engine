"""Focused tests for the strict YAML authoring contract."""

from decimal import Decimal
from pathlib import Path

import pytest

from spark_parser import (
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
    assert [column.data_type for column in config.columns] == [
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
    assert config.columns[3].parser.on_parse_error is ParseErrorMode.NULL
    assert config.columns[3].parser.audit is True


def test_safe_defaults_are_explicit() -> None:
    config = YamlParserConfigCompiler().compile_text(
        """
parser_config_id: simple
parser_config_name: Simple
version: "1"
columns:
  - column_name: raw_name
    data_type: string
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


@pytest.mark.parametrize(
    ("fragment", "message"),
    [
        ("data_type: decimal(18,32)\n    parser: decimal", "Decimal scale"),
        ("data_type: integer\n    parser: string", "incompatible"),
        (
            "data_type: integer\n    parser:\n      type: integer\n      is_nullable: false",
            "requires default_on_null",
        ),
        (
            "data_type: integer\n    parser:\n      type: integer\n"
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
  - column_name: value
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
  - column_name: amount
    data_type: decimal(8,2)
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
