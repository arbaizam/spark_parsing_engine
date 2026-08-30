"""Tests for public metadata discovery and configuration review reporting."""

from pathlib import Path

import pytest

from spark_parser import CompilationError, ParserType, SparkParserService, parser

TEST_CONFIG_PATH = Path(__file__).resolve().parents[1] / "fixtures" / "test_config.yaml"


def test_parser_metadata_is_discoverable_and_detached() -> None:
    description = parser.string.describe()

    assert description["parser_type"] == "string"
    assert "address_us_v1" in next(
        argument["allowed_values"]
        for argument in description["arguments"]
        if argument["name"] == "format"
    )
    format_values = next(
        argument["allowed_values"]
        for argument in description["arguments"]
        if argument["name"] == "format"
    )
    assert {"title", "state_us"} <= set(format_values)
    assert "null" in next(
        argument["allowed_values"]
        for argument in description["arguments"]
        if argument["name"] == "format"
    )
    string_error_modes = next(
        argument["allowed_values"]
        for argument in description["arguments"]
        if argument["name"] == "on_parse_error"
    )
    integer_error_modes = next(
        argument["allowed_values"]
        for argument in parser.integer.describe()["arguments"]
        if argument["name"] == "on_parse_error"
    )
    assert "preserve" in string_error_modes
    assert "preserve" not in integer_error_modes
    description["arguments"].clear()
    assert parser.string.describe()["arguments"]
    assert parser.config.describe()["column_arguments"][1]["name"] == "target_column_name"
    boolean_global = next(
        argument
        for argument in parser.config.describe()["global_arguments"]
        if argument["name"] == "boolean_case_sensitive"
    )
    assert boolean_global["default"] is False
    assert set(parser.describe()) == {member.value for member in ParserType}
    assert parser.array.describe()["parser_type"] == "array"
    assert parser.struct.describe()["parser_type"] == "struct"
    assert parser.map.describe()["parser_type"] == "map"
    array_child_modes = next(
        argument["allowed_values"]
        for argument in parser.array.describe()["arguments"]
        if argument["name"] == "on_element_error"
    )
    assert "preserve" in array_child_modes
    array_collapse = next(
        argument
        for argument in parser.array.describe()["arguments"]
        if argument["name"] == "collapse_whitespace"
    )
    assert array_collapse["default"] is False
    assert "whole container" in parser.map.describe()["gotchas"][0]
    assert parser.normalize_data_type("ARRAY < INT >") == "array<integer>"


def test_unknown_parser_description_uses_the_public_error_hierarchy() -> None:
    """Keep invalid authoring input catchable through SparkParserError/CompilationError."""
    with pytest.raises(CompilationError, match="Unknown parser type 'bogus'"):
        parser.describe("bogus")


def test_config_review_contains_validation_resolved_options_and_markdown(tmp_path: Path) -> None:
    service = SparkParserService()
    report = service.review_yaml(TEST_CONFIG_PATH)

    assert report.is_valid is True
    assert report.errors == ()
    assert report.summary["column_count"] == 4
    assert len(report.summary["content_hash"]) == 64
    assert report.column_reviews[0]["resolved_parser_options"]["collapse_whitespace"] is True
    assert report.column_reviews[0]["resolved_parser_options"]["default_on_error"] is None
    assert report.validation_checks[-1]["status"] == "N/A"
    assert "No Boolean parser nodes" in report.validation_checks[-1]["detail"]
    assert report.resolved_config is not None
    markdown = report.to_markdown()
    assert "Validation status:** PASS" in markdown
    assert "## Resolved parser options" in markdown
    assert "## Resolved globals" in markdown
    assert "## Resolved schema and parser tree" in markdown
    assert "## Canonical resolved configuration" in markdown
    assert "ColumnName1" in markdown
    recompiled = service.compile_mapping(report.resolved_config)
    assert service.content_hash(recompiled) == report.summary["content_hash"]
    markdown_path = report.write_markdown(tmp_path / "review.md")
    json_path = report.write_json(tmp_path / "review.json")
    assert markdown_path.read_text(encoding="utf-8").startswith("# Spark Parser")
    assert '"report_type": "spark_parser_config_review"' in json_path.read_text(
        encoding="utf-8"
    )
    assert report.to_mapping()["report_type"] == "spark_parser_config_review"


def test_invalid_config_review_returns_errors_instead_of_raising() -> None:
    report = parser.review_yaml("parser_config_id: incomplete")

    assert report.is_valid is False
    assert report.errors
    assert "columns" in report.errors[0]
    assert "Validation status:** FAIL" in report.to_markdown()


def test_missing_yaml_path_and_boolean_review_are_evidence_based() -> None:
    missing = parser.review_yaml("definitely_missing.yaml")
    assert missing.is_valid is False
    assert missing.errors == ("YAML file does not exist: definitely_missing.yaml",)

    boolean_report = parser.review_yaml(
        """
parser_config_id: boolean_review
parser_config_name: Boolean Review
version: "1"
columns:
  - source_column_name: active
    target_column_name: IsActive
    expected_data_type: boolean
    parser: boolean
"""
    )
    boolean_check = boolean_report.validation_checks[-1]
    assert boolean_check["status"] == "PASS"
    assert "1 Boolean parser node" in boolean_check["detail"]
