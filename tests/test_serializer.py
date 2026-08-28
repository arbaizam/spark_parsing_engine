"""Tests for deterministic config reporting metadata."""

from pathlib import Path

from spark_parser import ParserConfigSerializer, YamlParserConfigCompiler

ROOT = Path(__file__).resolve().parents[1]


def test_hash_and_mapping_are_deterministic() -> None:
    compiler = YamlParserConfigCompiler()
    serializer = ParserConfigSerializer()
    first = compiler.compile_path(ROOT / "test_config.yaml")
    second = compiler.compile_path(ROOT / "test_config.yaml")

    assert serializer.canonical_json(first) == serializer.canonical_json(second)
    assert serializer.content_hash(first) == serializer.content_hash(second)
    assert len(serializer.content_hash(first)) == 64
    payload = serializer.to_mapping(first)
    assert payload["columns"][0]["source_column_name"] == "column_name1"
    assert payload["columns"][0]["silver_column_name"] == "ColumnName1"
    assert payload["columns"][0]["expected_data_type"] == "string"
    assert payload["globals"]["true_values"] == ["true", "Y", "yes"]
    assert payload["columns"][0]["parser"]["trim_whitespace"] is True
    assert payload["columns"][0]["parser"]["collapse_whitespace"] is True
    assert payload["columns"][0]["parser"]["empty_is_null"] is True
    assert payload["columns"][0]["parser"]["audit"] is True
    recompiled = compiler.compile_mapping(payload)
    assert serializer.canonical_json(recompiled) == serializer.canonical_json(first)
