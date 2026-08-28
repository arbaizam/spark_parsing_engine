"""Tests for public metadata discovery and UAT review reporting."""

from pathlib import Path

from spark_parser import SparkParserService, parser

ROOT = Path(__file__).resolve().parents[1]


def test_parser_metadata_is_discoverable_and_detached() -> None:
    description = parser.string.describe()

    assert description["parser_type"] == "string"
    assert "address_us_v1" in next(
        argument["allowed_values"]
        for argument in description["arguments"]
        if argument["name"] == "format"
    )
    assert "null" in next(
        argument["allowed_values"]
        for argument in description["arguments"]
        if argument["name"] == "format"
    )
    description["arguments"].clear()
    assert parser.string.describe()["arguments"]
    assert parser.config.describe()["column_arguments"][1]["name"] == "silver_column_name"
    boolean_global = next(
        argument
        for argument in parser.config.describe()["global_arguments"]
        if argument["name"] == "boolean_case_sensitive"
    )
    assert boolean_global["default"] is False
    assert set(parser.describe()) == {
        "string",
        "integer",
        "long",
        "decimal",
        "double",
        "boolean",
        "date",
        "timestamp",
    }


def test_uat_report_contains_validation_resolved_options_and_markdown() -> None:
    service = SparkParserService()
    report = service.review_yaml(ROOT / "test_config.yaml")

    assert report.is_valid is True
    assert report.errors == ()
    assert report.summary["column_count"] == 4
    assert len(report.summary["content_hash"]) == 64
    assert report.column_reviews[0]["resolved_parser_options"]["collapse_whitespace"] is True
    assert report.column_reviews[0]["resolved_parser_options"]["default_on_error"] is None
    assert report.validation_checks[-1]["status"] == "N/A"
    assert "No Boolean mappings" in report.validation_checks[-1]["detail"]
    assert report.resolved_config is not None
    markdown = report.to_markdown()
    assert "Validation status:** PASS" in markdown
    assert "## Resolved parser options" in markdown
    assert "ColumnName1" in markdown
    assert report.to_mapping()["report_type"] == "spark_parser_uat_config_review"


def test_invalid_uat_report_returns_errors_instead_of_raising() -> None:
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
    silver_column_name: IsActive
    expected_data_type: boolean
    parser: boolean
"""
    )
    boolean_check = boolean_report.validation_checks[-1]
    assert boolean_check["status"] == "PASS"
    assert "1 Boolean mapping" in boolean_check["detail"]
