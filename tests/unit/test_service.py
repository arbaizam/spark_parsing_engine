"""Tests for public metadata discovery and configuration review reporting."""

import json
from pathlib import Path

import pytest

from spark_parser import (
    PARSER_DEFAULTS,
    CompilationError,
    ParserType,
    SparkParserService,
    YamlParserConfigCompiler,
    parser,
)
from spark_parser.defaults import (
    BUILTIN_DATETIME_FORMAT_SHAPES,
    DEFAULT_DATE_FORMATS,
    DEFAULT_TIMESTAMP_FORMATS,
    DEFAULT_TIMESTAMP_NTZ_FORMATS,
)

TEST_CONFIG_PATH = Path(__file__).resolve().parents[1] / "fixtures" / "test_config.yaml"


def test_public_defaults_are_deeply_immutable_and_detached_views_remain_mutable() -> None:
    """Make the constant immutable even through unbound built-in mutation methods."""
    with pytest.raises(TypeError):
        PARSER_DEFAULTS["common"]["trim_whitespace"] = False
    with pytest.raises(AttributeError):
        PARSER_DEFAULTS["date"]["formats"].append("yyyyMMdd")
    with pytest.raises(TypeError):
        dict.__setitem__(PARSER_DEFAULTS, "injected", {})
    with pytest.raises(TypeError):
        list.append(PARSER_DEFAULTS["globals"]["true_values"], "injected")

    detached = parser.defaults()
    detached["common"]["trim_whitespace"] = False
    detached["date"]["formats"].clear()

    assert PARSER_DEFAULTS["common"]["trim_whitespace"] is True
    assert PARSER_DEFAULTS["date"]["formats"] == DEFAULT_DATE_FORMATS
    assert json.loads(json.dumps(parser.defaults()))["date"]["formats"] == list(
        DEFAULT_DATE_FORMATS
    )
    assert parser.defaults()["common"]["trim_whitespace"] is True
    assert parser.defaults()["date"]["formats"] == list(DEFAULT_DATE_FORMATS)
    assert (
        next(
            argument["default"]
            for argument in parser.string.describe()["arguments"]
            if argument["name"] == "trim_whitespace"
        )
        is True
    )


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
    assert {
        "title",
        "title_business_v1",
        "interest_rate_index_v1",
        "property_type_v1",
        "state_us",
    } <= set(format_values)
    assert any("interest_rate_index_v1" in behavior for behavior in description["key_behaviors"])
    assert any("interest_rate_index_v1" in gotcha for gotcha in description["gotchas"])
    assert any("property_type_v1" in behavior for behavior in description["key_behaviors"])
    assert any("property_type_v1" in gotcha for gotcha in description["gotchas"])
    assert any("title_business_v1" in behavior for behavior in description["key_behaviors"])
    assert any("title_business_v1" in gotcha for gotcha in description["gotchas"])
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
    for floating_parser in (parser.float, parser.double):
        assert any("underflow" in gotcha for gotcha in floating_parser.describe()["gotchas"])
    description["arguments"].clear()
    assert parser.string.describe()["arguments"]
    assert parser.config.describe()["column_arguments"][1]["name"] == "target_column_name"
    source_argument = parser.config.describe()["column_arguments"][0]
    assert "fails binding unless" in source_argument["description"]
    boolean_global = next(
        argument
        for argument in parser.config.describe()["global_arguments"]
        if argument["name"] == "boolean_case_sensitive"
    )
    assert boolean_global["default"] is False
    assert set(parser.describe()) == {member.value for member in ParserType}
    assert parser.describe("numeric")["parser_type"] == "decimal"
    assert parser.describe("TIMESTAMP_LTZ")["parser_type"] == "timestamp"
    assert parser.normalize_data_type(" NUMERIC ( 18, 2 ) ") == "decimal(18,2)"
    with pytest.raises(CompilationError, match="Unsupported datatype"):
        parser.normalize_data_type("array<integer>")


def test_unknown_parser_description_uses_the_public_error_hierarchy() -> None:
    """Keep invalid authoring input catchable through SparkParserError/CompilationError."""
    with pytest.raises(CompilationError, match="Unknown parser type 'bogus'"):
        parser.describe("bogus")


def test_public_compile_and_serialization_facade_round_trips_scalar_schema() -> None:
    config = parser.compile_yaml(
        """
parser_config_id: facade
parser_config_name: Facade
version: "1"
columns:
  - source_column_name: raw_amount
    target_column_name: Amount
    expected_data_type: decimal(8,2)
    parser: decimal
"""
    )
    mapping = parser.to_mapping(config)
    canonical = parser.canonical_json(config)

    assert json.loads(canonical) == mapping
    assert parser.content_hash(parser.compile_mapping(mapping)) == parser.content_hash(config)
    report = parser.review_yaml(mapping)
    assert report.is_valid is True
    assert report.column_reviews[0]["schema_tree"].splitlines() == [
        '"Amount": decimal(8,2) [decimal] <- "raw_amount"',
    ]


def test_config_review_contains_validation_resolved_options_and_markdown(tmp_path: Path) -> None:
    service = SparkParserService()
    report = service.review_yaml(TEST_CONFIG_PATH)

    assert report.is_valid is True
    assert report.errors == ()
    assert report.summary["column_count"] == 4
    # Both fields remain available in the published report shape. Scalar-only configs have one
    # parser per column, so their counts are necessarily identical.
    assert report.summary["parser_node_count"] == report.summary["column_count"]
    assert len(report.summary["content_hash"]) == 64
    assert report.column_reviews[0]["resolved_parser_options"]["collapse_whitespace"] is True
    assert report.column_reviews[0]["resolved_parser_options"]["default_on_error"] is None
    assert report.validation_checks[-1]["status"] == "N/A"
    assert "No Boolean columns" in report.validation_checks[-1]["detail"]
    assert report.resolved_config is not None
    markdown = report.to_markdown()
    assert "Validation status:** PASS" in markdown
    assert "## Resolved parser options" in markdown
    assert "## Resolved globals" in markdown
    assert "## Resolved column bindings" in markdown
    assert "## Canonical resolved configuration" in markdown
    assert "ColumnName1" in markdown
    recompiled = service.compile_mapping(report.resolved_config)
    assert service.content_hash(recompiled) == report.summary["content_hash"]
    markdown_path = report.write_markdown(tmp_path / "review.md")
    json_path = report.write_json(tmp_path / "review.json")
    assert markdown_path.read_text(encoding="utf-8").startswith("# Spark Parser")
    assert '"report_type": "spark_parser_config_review"' in json_path.read_text(encoding="utf-8")
    detached = report.to_mapping()
    assert detached["report_type"] == "spark_parser_config_review"
    detached["summary"]["column_count"] = 0
    assert report.summary["column_count"] == 4


def test_markdown_review_contains_hostile_schema_names_inside_safe_fences() -> None:
    """Keep authored newlines and backticks from terminating generated code blocks."""
    report = SparkParserService().review_yaml(
        r"""
parser_config_id: markdown-safety
parser_config_name: Markdown safety
version: "1"
columns:
  - source_column_name: "raw\0\e\r<script>\n```\n## injected ![pixel](https://attacker.example/p) *name* a\\|b"
    target_column_name: "Safe\u202Egpj.exe"
    expected_data_type: string
    parser: string
"""
    )

    assert report.is_valid is True
    markdown = report.to_markdown()
    assert '"raw\\u0000\\u001b\\r<script>\\n```\\n## injected' in markdown
    assert "raw&#92;u0000&#92;u001b &#60;script&#62;" in markdown
    assert (
        "&#33;&#91;pixel&#93;&#40;https&#58;&#47;&#47;attacker&#46;example&#47;p&#41;" in markdown
    )
    assert "&#42;name&#42;" in markdown
    assert "a&#92;&#124;b" in markdown
    assert "\x00" not in markdown
    assert "\x1b" not in markdown
    assert "\u202e" not in markdown
    assert "\\u0000" in markdown
    assert "\\u001b" in markdown
    assert "\\u202e" in markdown
    assert "````text\n" in markdown
    assert "````yaml\n" in markdown
    assert markdown.count("````") == 4

    json_report = report.to_json()
    assert "\u202e" not in json_report
    assert "\\u202e" in json_report
    assert json.loads(json_report)["resolved_config"]["columns"][0]["target_column_name"] == (
        "Safe\u202egpj.exe"
    )

    yaml_fence = markdown.split("````yaml\n", 1)[1].split("\n````", 1)[0]
    recompiled = parser.compile_text(yaml_fence)
    assert parser.content_hash(recompiled) == report.summary["content_hash"]


def test_invalid_config_review_returns_errors_instead_of_raising() -> None:
    report = parser.review_yaml("parser_config_id: incomplete")

    assert report.is_valid is False
    assert report.source == "inline YAML"
    assert report.errors
    assert "columns" in report.errors[0]
    assert "Validation status:** FAIL" in report.to_markdown()


def test_source_dispatch_is_type_driven_and_boolean_review_is_evidence_based(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    missing_path = tmp_path / "definitely_missing.yaml"
    missing = parser.review_yaml(missing_path)
    assert missing.is_valid is False
    assert missing.source == str(missing_path)
    assert missing.errors[0].startswith(f"Unable to read parser config {missing_path}:")

    # Strings are always YAML text. File selection is explicit through Path or compile_path(), so
    # an inline scalar can never be reinterpreted because a same-named local file happens to exist.
    path_shaped_text = parser.review_yaml("definitely_missing.yaml")
    assert path_shaped_text.is_valid is False
    assert path_shaped_text.source == "inline YAML"
    assert path_shaped_text.errors == ("parser config must be a mapping.",)

    shadow_path = tmp_path / "shadow.yaml"
    shadow_path.write_text(TEST_CONFIG_PATH.read_text(encoding="utf-8"), encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    assert parser.review_yaml("shadow.yaml").source == "inline YAML"
    assert parser.review_yaml(Path("shadow.yaml")).is_valid is True

    compact = parser.compile_yaml(
        "{parser_config_id: compact, parser_config_name: Compact, version: '1', "
        "description: docs/config.txt, columns: [{source_column_name: value, "
        "target_column_name: Value, expected_data_type: string, parser: string}]}"
    )
    assert compact.description == "docs/config.txt"

    invalid_compact = parser.review_yaml(
        "{parser_config_id: compact, parser_config_name: Compact, version: '1', "
        "description: docs/config.txt}"
    )
    assert invalid_compact.source == "inline YAML"
    assert invalid_compact.errors == ("columns must be a non-empty list.",)

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
    assert "1 Boolean column" in boolean_check["detail"]

    unicode_report = parser.review_yaml(
        """
parser_config_id: unicode_boolean_review
parser_config_name: Unicode Boolean Review
version: "1"
globals:
  true_values: ["Ä"]
  false_values: ["ä"]
  boolean_case_sensitive: false
columns:
  - source_column_name: active
    target_column_name: IsActive
    expected_data_type: boolean
    parser: boolean
"""
    )
    unicode_check = unicode_report.validation_checks[-1]
    assert unicode_report.is_valid is True
    assert unicode_check["status"] == "DEFERRED"
    assert "Spark will validate non-ASCII case-insensitive overlap" in unicode_check["detail"]


def test_review_warns_when_global_null_markers_are_never_enabled() -> None:
    report = parser.review_yaml(
        """
parser_config_id: inert_markers
parser_config_name: Inert Markers
version: "1"
globals:
  null_markers: [NA, N/A]
columns:
  - source_column_name: value
    target_column_name: Value
    expected_data_type: string
    parser: string
"""
    )

    assert report.is_valid is True
    assert any("markers are inert" in warning for warning in report.warnings)


def test_compiler_keys_and_datetime_guards_match_published_contracts() -> None:
    compiler = YamlParserConfigCompiler()
    for parser_type in ParserType:
        allowed = compiler._parser_allowed_keys(parser_type)
        documented = {
            argument["name"] for argument in parser.describe(parser_type.value)["arguments"]
        }
        assert allowed == documented

    configured_formats = {
        *DEFAULT_DATE_FORMATS,
        *DEFAULT_TIMESTAMP_FORMATS,
        *DEFAULT_TIMESTAMP_NTZ_FORMATS,
    }
    assert configured_formats == set(BUILTIN_DATETIME_FORMAT_SHAPES)
