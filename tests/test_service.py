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
    description["arguments"].clear()
    assert parser.string.describe()["arguments"]
    assert parser.config.describe()["column_arguments"][1]["name"] == "silver_column_name"
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
