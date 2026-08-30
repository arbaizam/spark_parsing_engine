"""Tests for deterministic config reporting metadata."""

from pathlib import Path

from spark_parser import ParserConfigSerializer, YamlParserConfigCompiler

TEST_CONFIG_PATH = Path(__file__).resolve().parents[1] / "fixtures" / "test_config.yaml"


def test_hash_and_mapping_are_deterministic() -> None:
    compiler = YamlParserConfigCompiler()
    serializer = ParserConfigSerializer()
    first = compiler.compile_path(TEST_CONFIG_PATH)
    second = compiler.compile_path(TEST_CONFIG_PATH)

    assert serializer.canonical_json(first) == serializer.canonical_json(second)
    assert serializer.content_hash(first) == serializer.content_hash(second)
    assert len(serializer.content_hash(first)) == 64
    payload = serializer.to_mapping(first)
    assert payload["columns"][0]["source_column_name"] == "column_name1"
    assert payload["columns"][0]["target_column_name"] == "ColumnName1"
    assert payload["columns"][0]["expected_data_type"] == "string"
    assert payload["globals"]["true_values"] == ["true", "Y", "yes"]
    assert payload["columns"][0]["parser"]["trim_whitespace"] is True
    assert payload["columns"][0]["parser"]["collapse_whitespace"] is True
    assert payload["columns"][0]["parser"]["empty_is_null"] is True
    assert payload["columns"][0]["parser"]["audit"] is True
    recompiled = compiler.compile_mapping(payload)
    assert serializer.canonical_json(recompiled) == serializer.canonical_json(first)
    assert serializer.content_hash(recompiled) == serializer.content_hash(first)


def test_hash_tracks_behavior_and_order_but_not_yaml_layout() -> None:
    compiler = YamlParserConfigCompiler()
    serializer = ParserConfigSerializer()
    compact = compiler.compile_text(
        """
parser_config_id: hash_contract
parser_config_name: Hash Contract
version: "1"
columns:
  - {source_column_name: first, target_column_name: First, expected_data_type: string, parser: string}
  - {source_column_name: second, target_column_name: Second, expected_data_type: integer, parser: integer}
"""
    )
    expanded_layout = compiler.compile_text(
        """
version: "1"
parser_config_name: Hash Contract
parser_config_id: hash_contract
columns:
  - parser:
      type: string
    expected_data_type: string
    target_column_name: First
    source_column_name: first
  - expected_data_type: integer
    parser:
      type: integer
    source_column_name: second
    target_column_name: Second
"""
    )
    changed_behavior = compiler.compile_mapping(serializer.to_mapping(compact))
    changed_mapping = serializer.to_mapping(changed_behavior)
    changed_mapping["columns"][0]["parser"]["format"] = "upper"
    changed_behavior = compiler.compile_mapping(changed_mapping)
    reversed_mapping = serializer.to_mapping(compact)
    reversed_mapping["columns"].reverse()
    reversed_columns = compiler.compile_mapping(reversed_mapping)

    assert serializer.content_hash(compact) == serializer.content_hash(expanded_layout)
    assert serializer.content_hash(compact) != serializer.content_hash(changed_behavior)
    assert serializer.content_hash(compact) != serializer.content_hash(reversed_columns)
